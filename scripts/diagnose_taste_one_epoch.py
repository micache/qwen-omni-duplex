"""Controlled native-chat, duplex, and causal-parity probes of a TASTE adapter."""

import argparse
import json
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer

from duplex.contract import frame_event_inputs
from duplex.streaming import QwenDuplexStreamer
from duplex.training import MODEL_INPUTS, _load_adapter_weights, build_training_model, load_training_config
from duplex.turn_packed import NoiseAugmentationConfig, TurnPackedCollator, load_local_taste


GENERAL = [
    ('What is 17 + 25? Answer with the number only.', '42'),
    ('What is 9 times 7? Answer with the number only.', '63'),
    ('What is the capital of France? Answer with one word.', 'Paris'),
    ('What planet do humans live on? Answer with one word.', 'Earth'),
    ('Translate cat into Spanish. Answer with one word.', 'gato'),
    ('What is the opposite of hot? Answer with one word.', 'cold'),
    ('Which is larger, 12 or 19? Answer with the number only.', '19'),
    ('Continue this sequence with one number: 2, 4, 6, 8.', '10'),
    ('If all birds have wings and a robin is a bird, does a robin have wings? Answer yes or no.', 'yes'),
    ('Repeat exactly: The blue cup is on the table.', 'The blue cup is on the table.'),
    ('Sort these numbers ascending: 3, 1, 2. Answer as comma-separated numbers.', '1, 2, 3'),
    ('How many letters are in the word cat? Answer with the number only.', '3'),
]
SYSTEM = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'


def emit(output, row):
    with (output / 'probes.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def native(model, processor, prompt, reference, audio=None, *, max_new_tokens=64):
    content = prompt if audio is None else [{'type': 'audio', 'audio': 'local.wav'}]
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if audio is None:
        inputs = processor.tokenizer(text, return_tensors='pt').to('cuda')
    else:
        inputs = processor(text=text, audio=[audio], return_tensors='pt', padding=True)
        inputs = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
    ids = model.base_thinker.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      use_cache=True, eos_token_id=model.control_tokens.stop,
                                      pad_token_id=processor.tokenizer.pad_token_id)
    generated_ids = ids[0, inputs['input_ids'].shape[1]:].tolist()
    generated = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    result = {'text': generated, 'reference': reference,
              'generated_ids': generated_ids,
              'answer_prefix_correct': re.match(re.escape(reference.lower().rstrip('.')) + r'(?=$|[\s.,!?])', generated.strip().lower()) is not None,
              'normalized_exact': generated.strip().lower().rstrip('.') == reference.lower().rstrip('.')}
    if audio is None:
        reference_ids = processor.tokenizer(reference, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
        joined = torch.cat((inputs['input_ids'], reference_ids), dim=1)
        logits = model.base_thinker(input_ids=joined, attention_mask=torch.ones_like(joined), use_cache=False).logits
        start = inputs['input_ids'].shape[1] - 1
        scores = logits[:, start:start + reference_ids.shape[1]].float()
        result['reference_nll'] = float(F.cross_entropy(scores.reshape(-1, scores.shape[-1]), reference_ids.reshape(-1)))
    return result


def audio_vectors(model, batch):
    return model._restore_audio(
        input_features=batch['input_features'], feature_attention_mask=batch['feature_attention_mask'],
        preconv_feature_lengths=batch['preconv_feature_lengths'],
        audio_chunk_counts=batch.get('audio_chunk_counts'),
        current_attention=batch['attention_mask'], timeline_length=batch['text_ids'].shape[1],
        hidden_width=model.base_thinker.get_input_embeddings().weight.shape[1],
        device=torch.device('cuda'), dtype=torch.bfloat16)


def causal_probe(model, tokenizer, batch, timeline, audio, mode):
    """Identical audio and positions; optionally force gold controls or first token."""
    gold = batch['labels'][0].tolist()
    embedding = model.base_thinker.get_input_embeddings()
    cache, previous, selected_ids, details = None, None, [], []
    full_logits = model(**batch).logits[0] if mode == 'gold_history' else None
    lexical_count = 0
    for frame, target in enumerate(gold):
        text, tm, control, cm = frame_event_inputs(previous, bos_token_id=model.base_thinker.config.bos_token_id,
            control_ids=(model.control_tokens.idle, model.control_tokens.start, model.control_tokens.stop))
        fused = audio[:, frame:frame + 1] + embedding(torch.tensor([[text]], device='cuda')) * tm + embedding(torch.tensor([[control]], device='cuda')) * cm
        result = model.base_thinker.model(inputs_embeds=fused,
            attention_mask=torch.ones((1, frame + 1), dtype=torch.bool, device='cuda'),
            position_ids=torch.tensor([[frame]], device='cuda'), past_key_values=cache, use_cache=True, return_dict=True)
        cache = result.past_key_values
        logits = model.base_thinker.lm_head(result.last_hidden_state)[0, 0].float()
        is_text = timeline.targets.events[frame].kind.value == 'TEXT'
        if mode == 'gold_history':
            selected = target
            if is_text:
                top = logits.topk(5)
                details.append({'frame': frame, 'gold_id': target, 'gold': tokenizer.decode([target]),
                    'cached_argmax': int(logits.argmax()), 'full_argmax': int(full_logits[frame].argmax()),
                    'max_logit_difference': float((logits - full_logits[frame].float()).abs().max()),
                    'gold_probability': float(logits.softmax(-1)[target]),
                    'top': [{'id': int(i), 'token': tokenizer.decode([int(i)]), 'logit': float(v)} for i, v in zip(top.indices, top.values)]})
        elif not is_text or (mode == 'gold_first_token' and lexical_count == 0):
            selected = target
        else:
            logits[[model.control_tokens.idle, model.control_tokens.start, model.control_tokens.stop]] = -torch.inf
            selected = int(logits.argmax())
        if is_text:
            selected_ids.append(selected)
            lexical_count += 1
        previous = selected
        if target == model.control_tokens.stop:
            break
    return {'mode': mode, 'text': tokenizer.decode(selected_ids, clean_up_tokenization_spaces=False), 'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/turn_packed_one_epoch_a100.yaml')
    parser.add_argument('--output', type=Path, default=Path('outputs/one-epoch-diagnosis'))
    parser.add_argument('--adapter', action='append', type=Path)
    parser.add_argument('--dev-samples', type=int, default=32)
    parser.add_argument('--stream-samples', type=int, default=12)
    parser.add_argument('--compact', action='store_true', help='Only adapter stages, six general prompts, no audio/oracle probes.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'probes.jsonl').exists():
        raise FileExistsError('Use a fresh output folder to preserve previous probes.')
    config = load_training_config(args.config)
    model, processor, _ = build_training_model(config)
    model.eval()
    model.gradient_checkpointing_disable()
    collator = TurnPackedCollator(audio_processor=processor, audio_tower=model.base_thinker.audio_tower,
        tokenizer=processor.tokenizer, control_tokens=model.control_tokens,
        thinker_bos_token_id=model.base_thinker.config.bos_token_id, interruption_probability=0,
        min_assistant_frames=4, noise_dataset=None, noise_config=NoiseAugmentationConfig(), augmentation_seed=214)
    train = load_local_taste(config['data']['root'], split='train')
    dev = load_local_taste(config['data']['root'], split='dev')
    # First examples reproduce the reported failures; evenly spaced rows broaden the check.
    dev_indices = list(range(min(args.stream_samples, args.dev_samples)))
    dev_indices += np.linspace(args.stream_samples, len(dev) - 1, args.dev_samples - len(dev_indices), dtype=int).tolist()
    samples = [('train', i, train[i], train.dataset[i]['message'][0]['text']) for i in range(4)]
    samples += [('dev', i, dev[i], dev.dataset[i]['message'][0]['text']) for i in dev_indices]
    emit(args.output, {'probe': 'selection', 'dev_indices': dev_indices, 'train_indices': list(range(4)),
        'noise_probability': 0, 'interruption_probability': 0, 'context': [], 'general': GENERAL})
    adapters = args.adapter or [Path(config['training']['output_dir']) / 'checkpoint-1500', Path(config['training']['output_dir']) / 'final']
    started = time.perf_counter()
    stages = ([] if args.compact else [('base', None, 0)]) + [(str(p), p, 1) for p in adapters]
    if not args.compact:
        stages += [('final-scale-0.5', adapters[-1], 0.5), ('final-scale-0.25', adapters[-1], 0.25)]
    for label, checkpoint, scale in stages:
        if checkpoint is not None:
            _load_adapter_weights(model, checkpoint)
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.set_scale('default', scale)
        context = model.thinker.disable_adapter() if checkpoint is None else nullcontext()
        with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for prompt, reference in (GENERAL[:6] if args.compact else GENERAL):
                emit(args.output, {'stage': label, 'probe': 'native_general', 'prompt': prompt,
                    **native(model, processor, prompt, reference, max_new_tokens=16 if args.compact else 64)})
            for split, index, conversation, prompt in samples:
                if split == 'dev' and index < 3:
                    native_modes = [('native_text', None)]
                    if not args.compact:
                        native_modes.append(('native_audio', conversation.user_waveform))
                    for kind, waveform in native_modes:
                        emit(args.output, {'stage': label, 'probe': kind, 'sample': conversation.conversation_id,
                            'prompt': prompt, **native(model, processor, prompt, conversation.assistant_text, waveform,
                                max_new_tokens=16 if args.compact else 64)})
                if scale not in (0, 1) and (split != 'dev' or index >= 3):
                    continue
                timeline = collator._timeline(conversation)
                batch = {k: v.to('cuda') for k, v in collator([timeline]).items() if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
                result = model(**batch)
                gold = batch['labels'][0]
                mask = torch.tensor([e.kind.value == 'TEXT' for e in timeline.targets.events], device='cuda')
                predictions = result.logits[0].argmax(-1)
                emit(args.output, {'stage': label, 'probe': 'duplex_teacher_forced', 'split': split, 'index': index,
                    'sample': conversation.conversation_id, 'prompt': prompt, 'reference': conversation.assistant_text,
                    'user_seconds': len(conversation.user_waveform) / 16000,
                    'text_tokens': int(mask.sum()), 'frames': len(gold),
                    'text_correct': int((predictions[mask] == gold[mask]).sum()),
                    'first_token_correct': bool(predictions[mask][0] == gold[mask][0]),
                    'text_loss': float(result.group_losses['text']), 'weighted_loss': float(result.loss),
                    'predicted_text_at_gold_frames': processor.tokenizer.decode(predictions[mask].tolist())})
                del result
                if (split == 'train' or index < args.stream_samples) and checkpoint is not None:
                    streamer = QwenDuplexStreamer(model, processor, max_silent_chunks=4, max_new_tokens=64)
                    stream = streamer.run(conversation.user_waveform, sample_id=conversation.conversation_id, context_token_ids=())
                    starts = [r.audio_time_s for r in stream.trace if r.event_type == 'START']
                    emit(args.output, {'stage': label, 'probe': 'duplex_stream', 'split': split, 'index': index,
                        'sample': conversation.conversation_id, 'text': stream.text,
                        'exact': stream.text == conversation.assistant_text, 'timed_out': stream.timed_out,
                        'start_times': starts, 'user_end': len(conversation.user_waveform) / 16000,
                        'event_counts': dict(Counter(r.event_type for r in stream.trace))})
                if not args.compact and checkpoint == adapters[-1] and scale == 1 and split == 'dev' and index < 3:
                    audio = audio_vectors(model, batch)
                    for mode in ('gold_history', 'gold_controls', 'gold_first_token'):
                        emit(args.output, {'stage': label, 'probe': 'causal_oracle', 'sample': conversation.conversation_id,
                            **causal_probe(model, processor.tokenizer, batch, timeline, audio, mode)})
                    silent_audio = audio.clone()
                    boundary = next(i for i, e in enumerate(timeline.targets.events) if e.kind.value == 'START')
                    silent_audio[:, boundary:] = 0
                    emit(args.output, {'stage': label, 'probe': 'response_audio_zero', 'sample': conversation.conversation_id,
                        **causal_probe(model, processor.tokenizer, batch, timeline, silent_audio, 'gold_controls'),
                        'speech_feature_norm': float(audio[:, :boundary].float().norm(dim=-1).mean()),
                        'response_silence_feature_norm': float(audio[:, boundary:].float().norm(dim=-1).mean()),
                        'token_embedding_norm': float(model.base_thinker.get_input_embeddings()(batch['text_ids']).float().norm(dim=-1).mean())})
                del batch
        emit(args.output, {'stage': label, 'probe': 'stage_finished', 'elapsed_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
