Final experiment tools

Run these from assignment4 in its existing Python 3.12 virtual environment.

Current step is metadata checking only:
python -m py_compile final_retrieval.py final_generation.py freeze_gate.py prepare_freeze.py
python final_retrieval.py --check-only
python final_generation.py --check-config

The four Python files do not change the experiment configuration on import or in these check modes.
Actual final inference requires the prospective Protocol Freeze and a user-provided Moodle upload receipt.
The protocol document and analysis_plan.json are prepared in the next step before sealing.
prepare_freeze.py is for that later step; do not substitute an old protocol with different sample counts.

final_retrieval.py provides the 20-article, 2000-question retrieval stage.
final_generation.py provides 1000 questions per strategy, 3000 answers total, atomic checkpoints and resume.
freeze_gate.py validates local frozen artifacts and user-provided upload evidence, not Moodle server authenticity.
prepare_freeze.py provides separate prepare and register commands; preparing does not claim an upload.

Checks completed before delivery: syntax for all four files; metadata-only CLI checks on an isolated copy;
rejection of execution without upload evidence; five synthetic checkpoint/diagnostic checks.
No model inference or final predictions were run during these checks.
