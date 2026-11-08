"""Cache the exact public model revisions for a clean reproduction machine.

This downloads files only. It does not run embeddings or generate answers.
"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/'experiment_config.json').read_text())
    for kind in ['embedding','generation']:
        item=cfg[kind]
        print(f"Caching {item['name']} at revision {item['revision']}",flush=True)
        path=snapshot_download(repo_id=item['name'],revision=item['revision'],
            allow_patterns=['*.json','**/*.json','*.txt','*.model','model.safetensors'])
        print(path,flush=True)
    print('Pinned model files cached; no inference performed.')


if __name__=='__main__':main()
