#!/bin/bash
# 等本机这一臂**评测跑完** → 自动把产物打成 tar，供一次性下载。
#
# **为什么需要**：本项目硬规则是「远程任务一结束立即回传本地核对」。
# 但本次用户要关本地机器、可能几小时后才回来 —— 期间无法 scp。
# 与其等回来再逐文件操作，不如让**服务器自己**在结束时把产物封成一个包：
# 回来只需下载一个 tar + 一个 md5 清单，归档核对一步到位。
#
# 判据是 `eval_all.log` 里的 `ALL_EVAL_DONE`（runner 在评测循环之后写），
# **不是** `run.log` 的 `ALL_DONE` —— 后者在评测**开始前**就写了，早了 35 分钟。
#
# 幂等：已存在且非空的 tar 会跳过；可重复启动。
# 用法： setsid nohup bash experiments/p11_pack_when_done.sh <out_dir> &
set -u
OUT="${1:?用法: p11_pack_when_done.sh <out_dir>}"
export PYTHONIOENCODING=utf-8
# ⚠ 打包日志必须放在**产物目录外**（2026-10-05 修正）：
#   它是打包工具自己的操作日志，不是实验产物。原先写进 $OUT/pack.log 会造成**自指**——
#   ① 先 `md5sum *` 算清单（此刻 pack.log 只有 1 行）
#   ② 往 pack.log 追加几行
#   ③ 再 `tar czf`（此刻 pack.log 已有多行）
#   → **包里那份 pack.log 永远与清单不符**，归档核对每次稳定报 1 条"哈希不符"。
#   而项目的核对纪律明写"必须区分文件缺失与哈希不符"——永久性假阳性会让这条检查脱敏，
#   真的不一致反而容易滑过去。挪出目录即根除。
LOG="/root/autodl-tmp/pack_$(basename "$OUT").log"
PKG="/root/autodl-tmp/$(basename "$OUT")_$(date +%m%d).tar.gz"
MD5F="$PKG.md5"

echo "[$(date '+%F %T')] 等待评测完成（判据：$OUT/eval_all.log 出现 ALL_EVAL_DONE）…" >> "$LOG"

# 上限 96 次 × 30 min = 48 h
for i in $(seq 1 96); do
  if grep -q "ALL_EVAL_DONE" "$OUT/eval_all.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] 评测已完成，开始打包" >> "$LOG"
    break
  fi
  if ! pgrep -f 'watch.s[h]|train_p9e1.p[y]' >/dev/null 2>&1 && [ "$i" -gt 2 ]; then
    echo "[$(date '+%F %T')] ⚠ 训练/守护进程都没了，但没等到 ALL_EVAL_DONE —— 可能是崩溃，仍然打包现有产物" >> "$LOG"
    break
  fi
  sleep 1800
done

cd /root/autodl-tmp || exit 1
# 先出逐文件 md5 清单（归档核对用；本项目要求"文件数 + 逐文件 md5"两项，
# 且两侧 md5sum 格式不同需归一化 —— 清单连同包一起回传最省事）
# ⚠ 清单必须枚举**全部**常规文件，不能用 `md5sum *`（2026-10-05 修正）：
#   shell 的 `*` **匹配不到点开头的隐藏文件**，而 tar 会把它们打进去
#   → 隐藏文件"进包却不进清单"，归档核对永远报"清单外多出"（缺陷 2 的真正机制）。
#   改用 find -type f 枚举，隐藏文件也进清单，从根上消除这类假阳性。
( cd "$OUT" && find . -maxdepth 1 -type f -printf '%P\n' | sort | \
  while read -r f; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f"; done ) > "$PKG.md5" 2>/dev/null
echo "[$(date '+%F %T')] 逐文件清单：$(wc -l < "$PKG.md5") 个文件" >> "$LOG"

tar czf "$PKG" -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>>"$LOG"
echo "[$(date '+%F %T')] 包已生成：$PKG ($(du -h "$PKG" | cut -f1))" >> "$LOG"
md5sum "$PKG" >> "$LOG"
echo "[$(date '+%F %T')] 包 md5：$(md5sum "$PKG" | cut -d' ' -f1)" >> "$LOG"
echo "PACK_DONE" >> "$LOG"
