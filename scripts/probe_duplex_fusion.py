"""Measure native control bias and text sensitivity under additive fusion."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from diagnose_taste_one_epoch import GENERAL, SYSTEM, audio_vectors, emit
from duplex.contract import prompt_token_ids
from duplex.streaming import QwenDuplexStreamer
from duplex.training import MODEL_INPUTS, _load_adapter_weights, build_training_model, load_training_config
from duplex.turn_packed import NoiseAugmentationConfig, TurnPackedCollator, load_local_taste


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/one-epoch-fusion-probe'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'probes.jsonl').exists():
        raise FileExistsError('Use a fresh output folder.')
    config = load_training_config('configs/turn_packed_one_epoch_a100.yaml')
    model, processor, _ = build_training_model(config)
    _load_adapter_weights(model, Path(config['training']['output_dir']) / 'final')
    model.eval()
    model.gradient_checkpointing_disable()
    controls = [model.control_tokens.idle, model.control_tokens.start, model.control_tokens.stop]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for prompt, expected in GENERAL:
            text = processor.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = processor.tokenizer(text, return_tensors='pt').to('cuda')
            ids = model.base_thinker.generate(**inputs, max_new_tokens=32, min_new_tokens=1,
                suppress_tokens=controls[:2], eos_token_id=controls[2], do_sample=False, use_cache=True)
            generated = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            logits = model.base_thinker(**inputs, use_cache=False).logits[0, -1].float()
            masked = logits.clone()
            masked[controls] = -torch.inf
            reference_id = processor.tokenizer.encode(expected, add_special_tokens=False)[0]
            emit(args.output, {'probe': 'native_control_suppression', 'prompt': prompt, 'expected': expected,
                'text': generated, 'generated_ids': ids[0, inputs['input_ids'].shape[1]:].tolist(),
                'control_mass': float(logits.softmax(-1)[controls].sum()),
                'first_reference_nll': float(-logits.log_softmax(-1)[reference_id]),
                'masked_first_reference_nll': float(-masked.log_softmax(-1)[reference_id]),
                'masked_argmax': processor.tokenizer.decode([int(masked.argmax())]),
                'raw_argmax': processor.tokenizer.decode([int(logits.argmax())])})
        collator = TurnPackedCollator(audio_processor=processor, audio_tower=model.base_thinker.audio_tower,
            tokenizer=processor.tokenizer, control_tokens=model.control_tokens,
            thinker_bos_token_id=model.base_thinker.config.bos_token_id,
            interruption_probability=0, min_assistant_frames=4, noise_dataset=None,
            noise_config=NoiseAugmentationConfig(), augmentation_seed=214)
        dataset = load_local_taste(config['data']['root'], split='dev')
        totals = {'content': [0., 0, 0], 'separator': [0., 0, 0], 'first': [0., 0, 0], 'rest': [0., 0, 0]}
        original_restore = model._restore_audio
        for index in list(range(12)) + [221, 431, 641, 851, 1061, 1271, 1480, 1690, 1900, 2110, 2320, 2530, 2739, 2949, 3159, 3369, 3579, 3789, 3999]:
            conversation = dataset[index]
            if index < 3:
                stream = QwenDuplexStreamer(model, processor, max_silent_chunks=4, max_new_tokens=64).run(
                    conversation.user_waveform, sample_id=conversation.conversation_id,
                    context_token_ids=prompt_token_ids(processor.tokenizer))
                emit(args.output, {'probe': 'default_system_prefix', 'sample': conversation.conversation_id,
                    'text': stream.text, 'timed_out': stream.timed_out,
                    'events': [r.event_type for r in stream.trace if r.event_type != 'IDLE']})
            timeline = collator._timeline(conversation)
            batch = {k: v.to('cuda') for k, v in collator([timeline]).items() if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
            audio = audio_vectors(model, batch)
            lexical = [i for i, e in enumerate(timeline.targets.events) if e.kind.value == 'TEXT']
            for scale in ((1., .5, .1, .025, 0.) if index < 3 else (1.,)):
                calibrated = audio.clone()
                boundary = lexical[0] - 1
                calibrated[:, boundary:] *= scale
                model._restore_audio = lambda **kwargs: calibrated
                result = model(**batch)
                logits = result.logits[0].float()
                gold = batch['labels'][0]
                losses = F.cross_entropy(logits, gold, reduction='none')
                predictions = logits.argmax(-1)
                if scale == 1:
                    for offset, frame in enumerate(lexical):
                        token = processor.tokenizer.decode([int(gold[frame])])
                        for group in ('content' if re.search(r'[A-Za-z0-9]', token) else 'separator',
                                      'first' if offset == 0 else 'rest'):
                            totals[group][0] += float(losses[frame])
                            totals[group][1] += 1
                            totals[group][2] += int(predictions[frame] == gold[frame])
                if index < 3:
                    active_norm = float(model.base_thinker.get_input_embeddings()(batch['text_ids'])[0, lexical[1:]].float().norm(dim=-1).mean())
                    emit(args.output, {'probe': 'response_fusion_scale', 'sample': conversation.conversation_id,
                        'response_audio_scale': scale, 'text_loss': float(result.group_losses['text']),
                        'teacher_forced_predicted_text': processor.tokenizer.decode(predictions[lexical].tolist()),
                        'active_text_embedding_norm': active_norm,
                        'response_audio_norm': float(calibrated[:, lexical].float().norm(dim=-1).mean())})
                    if scale == 1:
                        altered = dict(batch)
                        altered['text_mask'] = torch.zeros_like(batch['text_mask'])
                        altered['text_mask'][:, 0] = 1  # Retain the required causal bootstrap.
                        without_text = model(**altered)
                        emit(args.output, {'probe': 'remove_previous_text', 'sample': conversation.conversation_id,
                            'text_loss': float(without_text.group_losses['text']),
                            'teacher_forced_predicted_text': processor.tokenizer.decode(without_text.logits[0, lexical].argmax(-1).tolist())})
                        del without_text
                del result
            model._restore_audio = original_restore
        emit(args.output, {'probe': 'token_category_summary', 'samples': 31,
            'groups': {k: {'tokens': v[1], 'mean_nll': v[0] / v[1], 'accuracy': v[2] / v[1]} for k, v in totals.items()}})


if __name__ == '__main__':
    main()
