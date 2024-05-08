# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Tuple
import base64
import io

import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from src.seamless_communication.inference import SequenceGeneratorOptions, Translator

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "sc_m4tv2"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`)"
        ),
        default="seamlessM4T_v2_large",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        help="Vocoder model name",
        default="vocoder_v2",
    )
    parser.add_argument(
        "--auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
        help="Worker auth key",
    )

    return parser


def set_generation_opts(
    job_data: Namespace,
) -> Tuple[SequenceGeneratorOptions, SequenceGeneratorOptions]:
    # Set text, unit generation opts.
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=job_data.get('text_generation_beam_size', 5),
        soft_max_seq_len=(
            job_data.get('text_generation_max_len_a', 1),
            job_data.get('text_generation_max_len_b', 200)
        ),
    )
    if job_data.get('text_unk_blocking', False):
        text_generation_opts.unk_penalty = torch.inf
    if job_data.get('text_generation_ngram_blocking', False):
        text_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=job_data.get('no_repeat_ngram_size', 4)
        )

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=job_data.get('unit_generation_beam_size', 5),
        soft_max_seq_len=(
            job_data.get('unit_generation_max_len_a', 25),
            job_data.get('unit_generation_max_len_b', 50)
        ),
    )
    if job_data.get('unit_generation_ngram_blocking', False):
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=job_data.get('no_repeat_ngram_size', 4)
        )
    return text_generation_opts, unit_generation_opts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using AIME-API"
    )
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the API server"
                        )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
                        )

    parser = add_inference_arguments(parser)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")
    translator = Translator(args.model_name, args.vocoder_name, device, dtype=dtype)
    if args.api_server:
        api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name())
        while True:
            job_data = api_worker.job_request()
            text_generation_opts, unit_generation_opts = set_generation_opts(job_data)

            logger.info(f"{text_generation_opts=}")
            logger.info(f"{unit_generation_opts=}")
            logger.info(
                f"unit_generation_ngram_filtering={job_data.get('unit_generation_ngram_filtering', False)}"
            )
            if job_data.get('audio_input'):
                task = 'S2ST' if job_data.get('generate_audio', False) else 'S2TT'
                task = 'ASR' if task == 'S2ST' and job_data.get('tgt_lang', 'eng') == job_data.get('src_lang', 'eng') else task
                base64_data = job_data.get('audio_input').split(',')[1]
                audio_data = base64.b64decode(base64_data)

                with io.BytesIO(audio_data) as buffer:
                    wav, sample_rate = torchaudio.load(buffer)
                    translator_input = torchaudio.functional.resample(
                        wav, orig_freq=sample_rate, new_freq=16_000
                    )
            else:
                task = 'T2ST' if job_data.get('generate_audio', False) else 'T2TT'
                translator_input = job_data.get('text_input')
            try:
                text_output, speech_output = translator.predict(
                    translator_input,
                    task,
                    job_data.get('tgt_lang', 'spa'),
                    src_lang=job_data.get('src_lang', 'eng'),
                    text_generation_opts=text_generation_opts,
                    unit_generation_opts=unit_generation_opts,
                    unit_generation_ngram_filtering=job_data.get('unit_generation_ngram_filtering', False)
                )
                output = {'text_output': str(text_output[0]), 'task': task, 'model_name': args.model_name}
                if speech_output is not None:
                    with io.BytesIO() as buffer:
                        torchaudio.save(
                            buffer,
                            speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                            format='wav',
                            sample_rate=speech_output.sample_rate,
                        )
                        output['audio_output'] = buffer
                        api_worker.send_job_results(output)
                else:                    
                    api_worker.send_job_results(output)
                logger.info(f"Translated text in {job_data.get('tgt_lang', 'spa')}: {str(text_output[0])}")
            except RuntimeError as error:
                logger.info(error)
                if 'The sequence generator returned no hypothesis at index 0' in str(error):
                    error = 'No audio input'
                api_worker.send_job_results({'error': str(error), 'task': task})

if __name__ == "__main__":
    main()
