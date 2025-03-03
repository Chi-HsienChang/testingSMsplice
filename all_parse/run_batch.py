import subprocess

# 9, 11, 17無法執行

for i in range(17, 1117):  # 1 到 1000
    print(f"Running index {i}...")
    subprocess.run(["python3", "runSMsplice_fba.py", str(i)], check=True)
