"""run-visible 演示：每秒打一行进度，跑 20 秒。验证无缓冲日志实时可见。"""
import time

for i in range(20):
    print(f"[{time.strftime('%H:%M:%S')}] tick {i+1}/20 ...", flush=True)
    time.sleep(1)
print("[done] 演示结束")
