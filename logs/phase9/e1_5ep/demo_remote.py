"""run-visible 远程演示：autodl 上每秒打一行，跑 10 秒。"""
import time

for i in range(10):
    print(f"[{time.strftime('%H:%M:%S')}] remote tick {i+1}/10", flush=True)
    time.sleep(1)
print("[remote done]")
