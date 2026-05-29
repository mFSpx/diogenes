#!/usr/bin/env python3
"""Tiny GOALS handoff helper; deliberately under 100 non-comment code lines."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREFIX='"Save This Prompt, Pass on this Handoff:"'; DEFAULT_ROOT=Path('GOALS')
README=("# GOALS\n\nCrash-recovery notes for active goal work.\n\n"
"Rules: read CURRENT_HANDOFF at goal start; update X/N after every step; append concise handoffs to GOAL_LOG; keep final goal step prefixed with " + PREFIX + ".\n\n"
"Yap Trap: this is where bot-yap gets compressed, not expanded. Current handoff should stay tiny: goal, step, completed, next action, resume command, evidence. No essays.\n\n"
"Files: GOAL_HANDOFF_PROMPT.md, CURRENT_HANDOFF.md, GOAL_LOG.md, GOAL_PROMPTS.md, AGENT_ORCHESTRATION_POLICY.md, EXTERNAL_PLUGIN_BUILD_MODE.md, MODEL_FABRIC_AUDIT.md, FOSS_REUSE_AUDIT.md, plugin_build_mode_bootstrap.json, DEMO_25_STEP_LOG.md, ARCHITECTURE_AUDIT.md. No nested folder sprawl.\n")
MODE=("# External Plugin Build Mode\n\nCodex-first, BYO-LLM: any CLI/agent/runtime can point at GOALS, read the handoff, then execute only the assigned coding slice.\n\nAdapter Contract: name, provider/local lane, endpoint/env key names only, model tier intent, command to dry-run, command to execute, expected receipt path, safety limits, and rollback/check command. Never store secret values here.\n\nWorkflow: operator slop is allowed in the incoming goal; orchestrator slop is not. The orchestrator translates vibes into small steps, picks cheapest capable agents/tools, writes explicit coding-only prompts, and records X/N handoffs.\n\nReuse rule: do not build a new coding agent, model gateway, or local model server here. Reuse FOSS tools such as LiteLLM, OpenCode, Aider, Continue, and llama.cpp when they fit; GOALS stays the tiny handoff/adapter contract layer.\n\nPass-on intent packet: use `scripts/goal_dev_control.py` for away-time windows and dev-supply routing. It records the operator time budget, current handoff cadence, effective LOC/hour, and the selected agent lane in a receipt under `05_OUTPUTS/goals/`.\n\nthree next pitches: (1) adapter registry JSON generated from existing scripts, (2) one dry-run router that chooses local/Groq/Cohere/llama.cpp by task tier, (3) optional CLI handoff packet exporter for Codex/Claude/generic CLI agents without changing GOALS law.\n")
POLICY=("# Agent Orchestration Policy\n\nCheapest Capable Model: do not change the main-window model from inside the agent unless a safe model-control tool exists and the operator asked for it.\n\nSubagents: spawn only for useful parallel, non-blocking work. Pick the smallest capable available model/tier: tiny/fast for grep, tests, repetitive code; mid for contained coding; strong/frontier only for architecture, security, or ambiguous cross-system reasoning. Verify current model names/prices when choosing real names; otherwise write tier intent, not fake model facts.\n\nPrompt contract: every subagent gets a coding-only prompt with file ownership, task, inputs, acceptance checks, no-revert warning, expected output, and concise handoff. Chunk work so one agent owns one clear slice. Sequence dependent tasks locally; parallelize only disjoint slices.\n\nSystem rule: use Dev Library, GOALS, status ledger, recovery_matrix, and existing workers before inventing new machinery. Zero daemon, zero background CPU, near-zero tokens unless a step actually needs a handoff.\n\nDev Supply Control: when the operator gives an away-time or open-ended build window, run `python3 scripts/goal_dev_control.py --away-minutes <minutes> --text \"<intent>\"` to compute cadence, effective LOC/hour, and cheapest-capable route. It uses existing deterministic hygiene + bandit primitives; no model calls, no daemon, no graph writes.\n\nSlop Control: prefer one existing home per rule. GOALS owns handoff/orchestration policy only; status facts go to STATUS_LEDGER; runtime proofs go to 05_OUTPUTS receipts; broad code complexity goes to `scripts/slop_audit_law.py`. Do not make new docs when an existing GOALS file or JSON manifest can hold the contract.\n\nCapability Preservation: least mutation wins. Do not remove, rename, disable, or narrow an existing system capability unless the operator explicitly asks or a receipt proves it is dead/superseded. Build center-out: improve the smallest shared spine first, then adapters; never sprawl sideways when an existing surface can hold the rule.\n\nArchitecture Authority Law: inventing new architecture is allowed only when the operator explicitly authorizes it with a mode bit, or when a blocker receipt proves the existing topology cannot satisfy the mission. The allowed mode bits are PATCH_MODE, BUGCHASE_MODE, THIN_ADAPTER_MODE, ARCHITECTURE_MODE, EXPERIMENT_MODE, and REPLACEMENT_MODE. If the current mode does not include an architecture-authorizing bit such as ARCHITECTURE_MODE, NEW_ARCHITECTURE_ALLOWED, EXPERIMENTAL_ARCHITECTURE, REPLACEMENT_ARCHITECTURE, or BUILD_NEW_TOPOLOGY, then new topology is forbidden unless a blocker receipt proves existing topology cannot work. New architecture must never be smuggled into a bugfix, cleanup, ingestion, test, or recovery task.\n\nNew Architecture Mode Contract: any explicitly authorized new architecture artifact must declare whether it is EXPERIMENTAL, CANDIDATE, ACTIVE_CANON, or REPLACEMENT; what it wraps, replaces, or extends; the promotion gate; the rollback plan; required receipts; and required tests. Experimental artifacts are isolated, candidate artifacts are not active canon, active canon requires explicit operator authorization, and replacement artifacts must include migration and deprecation/rollback proof. Treat `05_OUTPUTS/architecture_authority/new_architecture_<timestamp>.json` as the receipt surface for that contract.\n")
PROMPT=("# Goal Handoff Prompt\n\nCopy/paste this at the start of a goal, and append this as the final step of every goal plan.\n\n"+PREFIX+"\n\n"
"1. Read GOALS/CURRENT_HANDOFF.md if it exists.\n2. Write/update CURRENT_HANDOFF.md with goal, objective, current step 0/N, completed facts, next action, resume command.\n3. After every step, update X/N and append the same entry to GOAL_LOG.md.\n4. Technical Summary Review and Dev Notes: <=2 short sentences, plain engineer, tiny cryptid field-note flavor, minimal tokens.\n5. Yap Trap: no yappity-yap; move verbose reasoning to receipts/audits only when useful.\n6. Cheapest Capable Model: do not change the main-window model; subagents get the smallest capable available model/tier for their bounded coding task.\n\n"
"- [ ] "+PREFIX+" Update GOALS/CURRENT_HANDOFF.md and GOALS/GOAL_LOG.md with final N/N, verification evidence, next pickup action, and brief Technical Summary Review and Dev Notes.\n")

def utc_now()->str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def ensure(root:Path=DEFAULT_ROOT)->Path: root.mkdir(parents=True,exist_ok=True); return root

def build_handoff(*, goal_title:str, objective:str, step:int, total_steps:int, status:str, completed:str, next_action:str, resume_command:str, dev_notes:str, generated_at:str|None=None)->str:
    if total_steps<1: raise ValueError('total_steps must be >= 1')
    if step<0 or step>total_steps: raise ValueError('step must be between 0 and total_steps')
    generated_at=generated_at or utc_now()
    return f"# CURRENT GOAL HANDOFF\n\n{PREFIX}\n\n- Goal: {goal_title}\n- Generated: `{generated_at}`\n- Current step: {step}/{total_steps}\n- Status: {status}\n- Objective: {objective}\n- Completed: {completed}\n- Next action: {next_action}\n- Resume command: `{resume_command}`\n\nTechnical Summary Review and Dev Notes: {dev_notes}\n"

def append_goal_log(root:Path, *, step:int, total_steps:int, step_title:str, handoff:str)->None:
    with (root/'GOAL_LOG.md').open('a',encoding='utf-8') as f: f.write(f"\n---\n\n## Step {step}/{total_steps} — {step_title}\n\n{handoff.rstrip()}\n")

def update_step(*, root:Path=DEFAULT_ROOT, goal_title:str, objective:str, step:int, total_steps:int, step_title:str, status:str, completed:str, next_action:str, resume_command:str, dev_notes:str)->str:
    root=ensure(root); handoff=build_handoff(goal_title=goal_title,objective=objective,step=step,total_steps=total_steps,status=status,completed=completed,next_action=next_action,resume_command=resume_command,dev_notes=dev_notes)
    (root/'CURRENT_HANDOFF.md').write_text(handoff,encoding='utf-8'); append_goal_log(root,step=step,total_steps=total_steps,step_title=step_title,handoff=handoff); return handoff

def scaffold_workspace(root:Path=DEFAULT_ROOT)->dict[str,Any]:
    root=ensure(root); (root/'README.md').write_text(README,encoding='utf-8'); (root/'GOAL_HANDOFF_PROMPT.md').write_text(PROMPT,encoding='utf-8'); (root/'AGENT_ORCHESTRATION_POLICY.md').write_text(POLICY,encoding='utf-8'); (root/'EXTERNAL_PLUGIN_BUILD_MODE.md').write_text(MODE,encoding='utf-8')
    (root/'GOAL_PROMPTS.md').touch(); (root/'GOAL_LOG.md').touch()
    if not (root/'CURRENT_HANDOFF.md').exists():
        (root/'CURRENT_HANDOFF.md').write_text(build_handoff(goal_title='Blank Goal Handoff',objective='Waiting for the next submitted goal.',step=0,total_steps=1,status='blank',completed='Workspace scaffolded.',next_action='Paste a real goal and update this handoff before work starts.',resume_command='cat GOALS/CURRENT_HANDOFF.md',dev_notes='Empty trailhead ready; no cryptid tracks yet.'),encoding='utf-8')
    return check_workspace(root=root)

def demo_steps()->list[dict[str,str]]:
    first=[('Print Hello World','print','Hello World'),('Print We Do This Math Wrong','math','4+2=-1'),('Fill Cabbage Blank','blank','cabbage are the second stinkiest salmonoid'),('Teleportato a Kronenberg to a Borg','weird-route','Teleportato a Kronenberg to a Borg: route logged, no canonical mutation')]
    extras='Measure mothman wingnut torque|Index lake-serpent soccer cleats|Sort bog gremlin receipts|Checksum the haunted stapler|Classify sasquatch pocket lint|Ping the goblin build server|Name the raccoon theorem|Balance the chupacabra ledger|Render the swamp-light diagram|Fold the Jersey Devil napkin|Count vampire squash seeds|Audit the banshee lint trap|Reticulate kraken splines|Compile the yeti grocery list|Dry-run the phantom turnip|Route the kelpie bus pass|Normalize thunderbird shoelaces|Graph the haunted cabbage orbit|Smoke-test the mooncalf latch|Stage the ogopogo checksum|Seal the cryptid lab notebook'.split('|')
    steps=[{'title':t,'operation':o,'result':r} for t,o,r in first]
    steps += [{'title':t,'operation':'silly-op','result':f'step {i} logged; no side effects beyond GOALS notes'} for i,t in enumerate(extras,5)]; return steps

def run_demo(root:Path=DEFAULT_ROOT)->dict[str,Any]:
    root=ensure(root); scaffold_workspace(root); steps=demo_steps(); demo=root/'DEMO_25_STEP_LOG.md'; demo.write_text('# Demo 25-Step Goal Handoff Drill\n',encoding='utf-8')
    for i,item in enumerate(steps,1):
        note='Blank filled; cabbage taxonomy remains legally unserious.' if i==3 else ('Demo loop sealed; cryptid tracks are numbered and dry.' if i==25 else 'Logged the trail; tiny instruments, large footprints.')
        nxt=steps[i]['title'] if i<len(steps) else 'Use the handoff prompt on the next real goal.'
        h=update_step(root=root,goal_title='Demo 25-Step Goal Handoff Drill',objective='Prove step-by-step crash handoffs with silly operations.',step=i,total_steps=len(steps),step_title=item['title'],status='complete' if i==len(steps) else 'running',completed=f"Operation `{item['operation']}` produced: {item['result']}",next_action=nxt,resume_command='cat GOALS/CURRENT_HANDOFF.md',dev_notes=note)
        with demo.open('a',encoding='utf-8') as f: f.write(f"\n## Step {i}/{len(steps)} — {item['title']}\n\n- Operation: `{item['operation']}`\n- Result: {item['result']}\n\n{h.rstrip()}\n")
    report={'schema':'lucidota.goals.demo_report.v1','generated_at':utc_now(),'root':str(root),'step_count':len(steps),'demo_log':str(demo),'current_handoff':str(root/'CURRENT_HANDOFF.md'),'passed':len(steps)==25}
    (root/'DEMO_25_STEP_REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return report

def check_workspace(root:Path=DEFAULT_ROOT)->dict[str,Any]:
    root=Path(root); req=['README.md','GOAL_HANDOFF_PROMPT.md','AGENT_ORCHESTRATION_POLICY.md','EXTERNAL_PLUGIN_BUILD_MODE.md','CURRENT_HANDOFF.md','GOAL_LOG.md','GOAL_PROMPTS.md']; missing=[n for n in req if not (root/n).exists()]
    prompt=(root/'GOAL_HANDOFF_PROMPT.md').read_text(encoding='utf-8') if (root/'GOAL_HANDOFF_PROMPT.md').exists() else ''; current=(root/'CURRENT_HANDOFF.md').read_text(encoding='utf-8') if (root/'CURRENT_HANDOFF.md').exists() else ''
    nested=[p.name for p in root.iterdir() if p.is_dir()] if root.exists() else []; passed=root.exists() and not missing and PREFIX in prompt and 'append this as the final step' in prompt and (not current or (PREFIX in current and 'Current step:' in current)) and not nested
    return {'schema':'lucidota.goals.workspace_check.v1','generated_at':utc_now(),'root':str(root),'passed':passed,'missing':missing,'prompt_ok':PREFIX in prompt,'current_ok':not current or (PREFIX in current and 'Current step:' in current),'nested_dirs':nested}

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(); p.add_argument('--root',default=str(DEFAULT_ROOT)); sub=p.add_subparsers(dest='cmd',required=True)
    for c in 'scaffold demo check'.split(): sub.add_parser(c)
    s=sub.add_parser('step')
    for a in 'goal-title objective step total-steps step-title status completed next-action dev-notes'.split(): s.add_argument('--'+a,required=True)
    s.add_argument('--resume-command',default='cat GOALS/CURRENT_HANDOFF.md'); a=p.parse_args(argv); root=Path(a.root)
    r=scaffold_workspace(root) if a.cmd=='scaffold' else run_demo(root) if a.cmd=='demo' else check_workspace(root) if a.cmd=='check' else check_workspace(root)
    if a.cmd=='step': update_step(root=root,goal_title=a.goal_title,objective=a.objective,step=int(a.step),total_steps=int(a.total_steps),step_title=a.step_title,status=a.status,completed=a.completed,next_action=a.next_action,resume_command=a.resume_command,dev_notes=a.dev_notes); r=check_workspace(root)
    print(json.dumps(r,indent=2,sort_keys=True)); return 0 if r.get('passed') else 1
if __name__=='__main__': raise SystemExit(main())
