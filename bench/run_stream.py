# SPDX-License-Identifier: MIT
"""Single-stream fixed-config smoke/benchmark with durable raw SSE evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def summarize(events, wall_seconds):
    token_events = []
    usage = None
    for event in events:
        data = event['data']
        if data.get('usage'):
            usage = data['usage']
        for choice in data.get('choices', []):
            if choice.get('token_ids'):
                token_events.append((event['seconds'], choice))
    if not token_events:
        raise ValueError('No emitted token IDs; cannot claim exact token throughput')
    total = sum(len(c['token_ids']) for _, c in token_events)
    interval = token_events[-1][0] - token_events[0][0]
    numerator = total - len(token_events[0][1]['token_ids'])
    channels = {'reasoning': '', 'answer': ''}
    after_first = {'reasoning': '', 'answer': ''}
    for event in events:
        for choice in event['data'].get('choices', []):
            delta = choice.get('delta', {})
            texts = {'reasoning': delta.get('reasoning') or delta.get('reasoning_content') or '',
                     'answer': delta.get('content') or choice.get('text') or ''}
            for key, value in texts.items():
                channels[key] += value
                if event['seconds'] > token_events[0][0]:
                    after_first[key] += value
    if usage and usage['completion_tokens'] != total:
        raise ValueError(f'Token ID/usage mismatch: {total} vs {usage}')
    return {'completion_tokens': total, 'usage': usage,
            'ttft_seconds': token_events[0][0], 'decode_seconds': interval,
            'decode_tokens_after_first_event': numerator,
            'first_event_tokens': len(token_events[0][1]['token_ids']),
            'decode_tok_per_second': numerator / interval if interval > 0 else None,
            'end_to_end_tok_per_second': total / wall_seconds,
            'character_count': {k: len(v) for k, v in channels.items()},
            'characters_per_decode_second': {k: len(v)/interval if interval > 0 else None
                                              for k, v in after_first.items()},
            'output': channels, 'timing_scope': 'client arrival timestamps; tokens per SSE chunk',
            'speculation_counters': None}


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-id')
    parser.add_argument('--run-kind',default='phase1_single_smoke')
    parser.add_argument('--warmup',action='store_true')
    parser.add_argument('--metrics',action='store_true')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a run directory')
    args.output.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    prompts_path = args.config.parent.parent / config['prompts']
    prompts = json.loads(prompts_path.read_text())
    if args.prompt_id:
        prompts = [p for p in prompts if p['id'] == args.prompt_id]
        if not prompts:
            raise ValueError('Unknown prompt ID')
    for prompt in prompts:
        request = {'model': 'glm53-exl3', 'messages': [{'role': 'user', 'content': prompt['prompt']}],
                   'temperature': config['temperature'], 'top_p': config['top_p'],
                   'seed': config['seed'], 'min_tokens': config['min_tokens'],
                   'max_tokens': config['max_tokens'], 'ignore_eos': config['ignore_eos'],
                   **config['thinking']['request_encoding'], 'stream': True,
                   'stream_options': {'include_usage': True}, 'return_token_ids': True}
        raw_path = args.output/(prompt['id']+'.jsonl')
        events = []
        metrics_before=None
        if args.metrics:
            with urllib.request.urlopen(args.url+'/metrics',timeout=10) as response:
                metrics_before=response.read().decode()
        started = time.perf_counter()
        result = {'prompt': prompt, 'request': request,
                  'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  'utc': datetime.now(timezone.utc).isoformat(), 'run_kind': args.run_kind,
                  'warmup': args.warmup, 'passed': False}
        try:
            req = urllib.request.Request(args.url+'/v1/chat/completions',
                                         data=json.dumps(request).encode(),
                                         headers={'Content-Type': 'application/json'})
            with raw_path.open('x') as raw, urllib.request.urlopen(req, timeout=1800) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    payload = line[6:].strip()
                    if payload == b'[DONE]':
                        break
                    event = {'seconds': time.perf_counter()-started, 'data': json.loads(payload)}
                    raw.write(json.dumps(event, ensure_ascii=False)+'\n')
                    raw.flush()
                    events.append(event)
            result.update(summarize(events, time.perf_counter()-started))
            if args.metrics:
                # Save counters outside the timed request; delayed engine metrics
                # are preserved rather than silently interpreted as step records.
                time.sleep(1)
                with urllib.request.urlopen(args.url+'/metrics',timeout=10) as response:
                    result['metrics_after']=response.read().decode()
                result['metrics_before']=metrics_before
            result['passed'] = result['completion_tokens'] == config['output_tokens']
            if not result['passed']:
                raise ValueError('Output length differs from frozen configuration')
        except Exception as exc:
            result['error'] = repr(exc)
            raise
        finally:
            (args.output/(prompt['id']+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ('completion_tokens', 'decode_tok_per_second', 'passed')}), flush=True)


if __name__ == '__main__':
    main()
