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

from seamless_communication.inference import SequenceGeneratorOptions, Translator

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "sc_m4tv2"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, help="Task type")
    parser.add_argument(
        "--tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language, only required if input is text.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the generated audio.",
        default=None,
    )
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
    # Text generation args.
    parser.add_argument(
        "--text_generation_beam_size",
        type=int,
        help="Beam size for incremental text decoding.",
        default=5,
    )
    parser.add_argument(
        "--text_generation_max_len_a",
        type=int,
        help="`a` in `ax + b` for incremental text decoding.",
        default=1,
    )
    parser.add_argument(
        "--text_generation_max_len_b",
        type=int,
        help="`b` in `ax + b` for incremental text decoding.",
        default=200,
    )
    parser.add_argument(
        "--text_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental text decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        help="Size of ngram repeat block for both text & unit decoding.",
        default=4,
    )
    # Unit generation args.
    parser.add_argument(
        "--unit_generation_beam_size",
        type=int,
        help=(
            "Beam size for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=5,
    )
    parser.add_argument(
        "--unit_generation_max_len_a",
        type=int,
        help=(
            "`a` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=25,
    )
    parser.add_argument(
        "--unit_generation_max_len_b",
        type=int,
        help=(
            "`b` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=50,
    )
    parser.add_argument(
        "--unit_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental unit decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--unit_generation_ngram_filtering",
        type=bool,
        help=(
            "If True, removes consecutive repeated ngrams"
            "from the decoded unit output."
        ),
        default=False,
    )
    parser.add_argument(
        "--text_unk_blocking",
        type=bool,
        help=(
            "If True, set penalty of UNK to inf in text generator "
            "to block unk output."
        ),
        default=False,
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
        description="M4T inference on supported tasks using Translator."
    )
    parser.add_argument("--input", type=str, help="Audio WAV file path or text input.", required=False)
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
        api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name())
        while True:
            job_data = api_worker.job_request()
            """
            if not args.task or not args.tgt_lang:
                raise Exception(
                    "Please provide required arguments for evaluation -  task, tgt_lang"
                )

            if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
                raise ValueError("output_path must be provided to save the generated audio")
            """
            text_generation_opts, unit_generation_opts = set_generation_opts(job_data)

            logger.info(f"{text_generation_opts=}")
            logger.info(f"{unit_generation_opts=}")
            logger.info(
                f"unit_generation_ngram_filtering={job_data.get('unit_generation_ngram_filtering', False)}"
            )

            # If the input is audio, resample to 16kHz

            if job_data.get('audio_input'):#, 'S2ST').upper() in {"S2ST", "ASR", "S2TT"}:
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
            print('task: ', task)
            text_output, speech_output = translator.predict(
                translator_input,
                task,
                job_data.get('tgt_lang', 'spa'),
                src_lang=job_data.get('src_lang', 'eng'),
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                unit_generation_ngram_filtering=job_data.get('unit_generation_ngram_filtering', False)
            )
            
            if speech_output is not None:
                with io.BytesIO() as buffer:
                    #logger.info(f"Saving translated audio in {job_data.get('tgt_lang', 'spa')}")
                    torchaudio.save(
                        buffer,
                        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                        format='wav',
                        sample_rate=speech_output.sample_rate,
                    )
                    api_worker.send_job_results({'text_output': str(text_output[0]), 'audio_output': buffer})
            else:
                api_worker.send_job_results({'text_output': str(text_output[0])})


            logger.info(f"Translated text in {job_data.get('tgt_lang', 'spa')}: {text_output[0]}")

def convert_base64_string_to_audio(base64_string):
    base64_data = base64_string.split(',')[1]
    audio_data = base64.b64decode(base64_data)

    with io.BytesIO(audio_data) as buffer:
        image = convert_img(Image.open(buffer))
    return image



if __name__ == "__main__":
    main()
