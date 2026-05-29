import os

file_path = "scripts/updated_abcd_sequence_runner.py"
with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for line in lines:
        f.write(line)
        if "COUNTED pillar_" in line or "COUNTED" in line and "epistemic=FACT" in line:
            # Inject the generation hook directly after the ledger update
            f.write("        # [INJECTED OROBOROS HOOK]\n")
            f.write("        if 'pillar_' in work_item_name:\n")
            f.write("            print(f'>>> [LLM SPARK] IGNITING AUTONOMOUS GENERATION FOR {work_item_name}')\n")
            f.write("            os.system(f'python3 -c \"print(\\'[MOCK GENERATION: WRITING CODE FOR {work_item_name}...]\n[MOCK COMPILING...]\\')\"')\n") # Replace with your actual agent CLI trigger
            f.write("            os.system(f'python3 -m pytest math/ -v')\n")
