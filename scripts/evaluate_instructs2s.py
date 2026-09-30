"""Check audio parity and evaluate complete answers without inference text prompts."""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch

from duplex.instructs2s import InstructS2SFirstTurnDataset
from duplex.batching import FrameBudgetBatchSampler
from torch.utils.data import DataLoader
from duplex.streaming import QwenDuplexStreamer, pad_audio_to_chunk_boundary, split_fixed_audio_chunks, write_trace_jsonl
from duplex.training import MODEL_INPUTS, _load_adapter_weights, build_training_model, load_training_config, seed_everything
from duplex.turn_packed import NoiseAugmentationConfig, PackedConversation, TurnPackedCollator
from duplex.turn_packed import decode_audio
from diagnose_taste_one_epoch import GENERAL, audio_vectors, causal_probe, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/instructs2s_one_epoch_a100.yaml')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--skip-native', action='store_true')
    parser.add_argument('--native-audio', action='store_true')
    parser.add_argument('--all-dev-teacher', action='store_true')
    parser.add_argument('--short-audio-probes', action='store_true')
    parser.add_argument('--check-parity', action='store_true')
    parser.add_argument('--samples-per-split', type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Preserve existing results; use a fresh report directory.')
    config = load_training_config(args.config)
    seed_everything(config['training']['seed'])
    started = time.monotonic()
    model, processor, _ = build_training_model(config)
    if args.adapter:
        _load_adapter_weights(model, args.adapter)
    model.eval()
    model.gradient_checkpointing_disable()
    collator = TurnPackedCollator(audio_processor=processor, audio_tower=model.base_thinker.audio_tower,
        tokenizer=processor.tokenizer, control_tokens=model.control_tokens,
        thinker_bos_token_id=model.base_thinker.config.bos_token_id,
        interruption_probability=0, noise_dataset=None, noise_config=NoiseAugmentationConfig(),
        min_assistant_frames=4, augmentation_seed=214)
    streamer = QwenDuplexStreamer(model, processor, sample=False)
    root = Path(config['data']['root'])
    samples = []
    if (root / 'train.jsonl').exists():
        for split in ('train', 'dev'):
            dataset = InstructS2SFirstTurnDataset(root, split=split)
            for index in range(min(args.samples_per_split, len(dataset))):
                samples.append((split, index, dataset.records[index], dataset[index]))
    elif args.preflight:
        rows = json.loads((root / 'prepared-shards/0.json').read_text())['rows']
        for index, row in enumerate(rows[:3]):
            raw, rate = sf.read(root / row['audio'], dtype='float32')
            assert rate == 16000
            conversation = PackedConversation(row['id'], pad_audio_to_chunk_boundary(raw),
                row['assistant_samples'], row['answer'], user_valid_samples=len(raw))
            samples.append(('preflight', index, row, conversation))
    else:
        raise FileNotFoundError('Dataset preparation has not finished.')
    if args.short_audio_probes:
        import pyarrow.parquet as pq
        # Lock the selection before observing generations. These short-answer
        # questions are excluded by the training subset's 16-token minimum.
        selected_ids = ['instruct_en_11', 'instruct_en_20', 'instruct_en_56',
                        'instruct_en_92', 'instruct_en_93', 'instruct_en_121']
        used_rows = [row for split in ('train', 'dev') for row in
                     InstructS2SFirstTurnDataset(root, split=split).records]
        used = {row['id'] for row in used_rows}
        used_prompts = {' '.join(row['prompt'].casefold().split()) for row in used_rows}
        assert not used.intersection(selected_ids)
        source = pq.read_table('outputs/instructs2s-inspection/part-0.parquet',
            columns=['id', 'round', 'question', 'answer', 'question_audio']).to_pylist()
        rows = {row['id']: row for row in source if row['round'] == 1 and row['id'] in selected_ids}
        for index, sample_id in enumerate(selected_ids):
            source_row = rows[sample_id]
            prompt = source_row['question'].removeprefix('<USER>:').strip()
            assert ' '.join(prompt.casefold().split()) not in used_prompts
            waveform = decode_audio(source_row['question_audio'], field='question_audio')
            path = (args.output / 'input-audio' / f'{sample_id}.flac').resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, waveform, 16000, subtype='PCM_16')
            waveform, _ = sf.read(path, dtype='float32')
            text = source_row['answer'].strip()
            tokens = processor.tokenizer.encode(text, add_special_tokens=False)
            assistant_samples = math.ceil((len(tokens)+2)/50)*32000
            row = {'id': sample_id, 'prompt': prompt,
                'answer': text, 'response_tokens': len(tokens), 'audio': str(path),
                'user_samples': len(waveform), 'assistant_samples': assistant_samples}
            conversation = PackedConversation(sample_id, pad_audio_to_chunk_boundary(waveform),
                assistant_samples, text, user_valid_samples=len(waveform))
            samples.append(('short_audio_ood', index, row, conversation))
    report = {'adapter': str(args.adapter) if args.adapter else None, 'context_token_ids': [],
              'selection': 'first rows of each split; no cherry picking', 'cases': [], 'native_text': []}
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for prompt, reference in ([] if args.skip_native else GENERAL):
            result = native(model, processor, prompt, reference, max_new_tokens=32)
            report['native_text'].append({'prompt': prompt, **result})
            print(json.dumps({'native_text': report['native_text'][-1]}, ensure_ascii=False), flush=True)
            save()
        for split, index, row, conversation in samples:
            timeline = collator._timeline(conversation)
            cpu_batch = collator([timeline])
            batch = {k: v.to('cuda') for k, v in cpu_batch.items()
                     if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
            prediction = model(**batch)
            gold = batch['labels'][0]
            lexical = torch.tensor([e.kind.value == 'TEXT' for e in timeline.targets.events], device='cuda')
            case = {'split': split, 'index': index, 'sample_id': row['id'], 'prompt': row['prompt'],
                'reference_text': row['answer'], 'response_tokens': row['response_tokens'],
                'user_duration_s': row['user_samples'] / 16000,
                'teacher_forced': {'weighted_loss': float(prediction.loss),
                    'group_losses': {k: float(v) for k, v in prediction.group_losses.items()},
                    'lexical_token_accuracy': float((prediction.logits[0].argmax(-1)[lexical] == gold[lexical]).float().mean())}}
            del prediction
            if args.native_audio:
                raw, _ = sf.read(root / row['audio'], dtype='float32')
                case['native_audio'] = native(model, processor, '', row['answer'], audio=raw, max_new_tokens=128)
            if (args.preflight or args.check_parity) and index < 3:
                training_audio = audio_vectors(model, batch)
                chunks = split_fixed_audio_chunks(conversation.input_waveform)
                maximum_mel_difference = 0.0
                maximum_embedding_difference = 0.0
                maximum_relative_rms = 0.0
                for chunk_index, chunk in enumerate(chunks):
                    processed = processor.feature_extractor([chunk.waveform], sampling_rate=16000,
                        padding=True, return_attention_mask=True, return_tensors='pt')
                    diff = float((cpu_batch['input_features'][chunk_index] - processed['input_features'][0]).abs().max())
                    maximum_mel_difference = max(maximum_mel_difference, diff)
                    encoded = streamer._audio_features(chunk)
                    reference_audio = training_audio[:, chunk_index*50:(chunk_index+1)*50]
                    maximum_embedding_difference = max(maximum_embedding_difference, float((encoded-reference_audio).abs().max()))
                    relative_rms = float((encoded.float()-reference_audio.float()).square().mean().sqrt()
                                         / reference_audio.float().square().mean().sqrt())
                    maximum_relative_rms = max(maximum_relative_rms, relative_rms)
                assert maximum_mel_difference <= 1e-5
                assert batch['context_ids'].shape[-1] == 0
                # Separate BF16 batch-size rounding from structural input mismatches.
                # The same frozen weights are evaluated in FP32 solely for this check.
                maximum_fp32_difference = 0.0
                model.base_thinker.audio_tower.float()
                with torch.autocast('cuda', enabled=False):
                    full = model.base_thinker.get_audio_features(input_features=batch['input_features'].float(),
                        feature_attention_mask=batch['feature_attention_mask'], return_dict=True).last_hidden_state
                    for chunk_index in range(len(chunks)):
                        single = model.base_thinker.get_audio_features(
                            input_features=batch['input_features'][chunk_index:chunk_index+1].float(),
                            feature_attention_mask=batch['feature_attention_mask'][chunk_index:chunk_index+1],
                            return_dict=True).last_hidden_state
                        reference_audio = full[chunk_index*50:(chunk_index+1)*50]
                        torch.testing.assert_close(single, reference_audio, atol=0.003, rtol=0.0001)
                        maximum_fp32_difference = max(maximum_fp32_difference, float((single-reference_audio).abs().max()))
                if not getattr(model, 'audio_encoder_fp32', False):
                    model.base_thinker.audio_tower.bfloat16()
                del full, single
                case['audio_parity'] = {'chunks': len(chunks), 'mel_max_abs_difference': maximum_mel_difference,
                    'embedding_max_abs_difference_after_bf16_fusion_cast': maximum_embedding_difference,
                    'embedding_max_relative_rms_after_bf16_fusion_cast': maximum_relative_rms,
                    'embedding_max_abs_difference_fp32': maximum_fp32_difference,
                    'audio_frame_norm_median': float(training_audio.float().norm(dim=-1).median()),
                    'text_token_embedding_norm_median': float(model.base_thinker.get_input_embeddings()(gold[lexical]).float().norm(dim=-1).median()),
                    'full_chunk_masks': bool((cpu_batch['preconv_feature_lengths'] == 200).all())}
                if index == 0:
                    cached = causal_probe(model, processor.tokenizer, batch, timeline, training_audio, 'gold_history')
                    case['cached_gold_history'] = {'lexical_positions': len(cached['details']),
                        'argmax_matches': sum(r['cached_argmax'] == r['full_argmax'] for r in cached['details']),
                        'max_logit_difference': max(r['max_logit_difference'] for r in cached['details'])}
                del training_audio
            del batch, cpu_batch
            if not args.preflight:
                # The model receives only this real question clip, never reference response padding.
                raw, rate = sf.read(root / row['audio'], dtype='float32')
                result = streamer.run(raw, sample_id=f'{split}-{row["id"]}', context_token_ids=(), sample_rate_hz=rate)
                trace = args.output / 'traces' / f'{split}-{index}.jsonl'
                write_trace_jsonl(trace, result.trace)
                case['stream'] = {'text': result.text, 'timed_out': result.timed_out,
                    'stop_reason': result.stop_reason, 'event_counts': dict(Counter(r.event_type for r in result.trace)),
                    'exact_reference': result.text == row['answer'], 'trace': str(trace),
                    'grammar_mask_changed_count': sum(r.grammar_mask_changed_raw_argmax for r in result.trace)}
            report['cases'].append(case)
            print(json.dumps(case, ensure_ascii=False), flush=True)
            save()
    if args.all_dev_teacher:
        dataset = InstructS2SFirstTurnDataset(root, split='dev')
        sampler = FrameBudgetBatchSampler(dataset.frame_lengths, max_frames=config['training']['max_batch_frames'],
                                          max_batch_size=16, seed=212)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator,
                            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        counts = Counter()
        losses = Counter()
        correct = 0
        weight_sum = loss_sum = 0.0
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for index, cpu_batch in enumerate(loader):
                batch = {k: v.to('cuda', non_blocking=True) for k, v in cpu_batch.items()
                         if k in MODEL_INPUTS and isinstance(v, torch.Tensor)}
                prediction = model(**batch)
                for group in ('text', 'idle', 'start', 'stop'):
                    count = int(prediction.target_counts[group])
                    counts[group] += count
                    losses[group] += float(prediction.group_losses[group]) * count
                labels = batch['labels']
                lexical = (labels != -100) & (labels != model.control_tokens.idle) & (labels != model.control_tokens.start) & (labels != model.control_tokens.stop)
                correct += int(((prediction.logits.argmax(-1) == labels) & lexical).sum())
                weight = float(prediction.loss_weight_sum)
                weight_sum += weight
                loss_sum += float(prediction.loss) * weight
                del batch, prediction, cpu_batch
                if (index+1) % 25 == 0:
                    print(json.dumps({'clean_dev_batches_completed': index+1, 'total_batches': len(sampler)}), flush=True)
        report['clean_dev_teacher_forced'] = {'conversations': len(dataset), 'target_counts': dict(counts),
            'group_losses': {group: losses[group]/counts[group] for group in counts},
            'weighted_loss': loss_sum/weight_sum, 'lexical_token_accuracy': correct/counts['text'],
            'noise_probability': 0, 'interruption_probability': 0}
        print(json.dumps({'clean_dev_teacher_forced': report['clean_dev_teacher_forced']}, indent=2), flush=True)
    save()


if __name__ == '__main__':
    main()
