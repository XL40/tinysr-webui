#!/usr/bin/env bash
# 拉取构建依赖 (ncnn 源码 + 单头库)。ncnn 与 include 下的头文件不入库, 需本地执行一次。
# 所有版本钉死到 commit / tag, 保证与开发时验证过的构建组合完全一致。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party/include

# 可选: 走代理导出 HTTPS_PROXY=http://127.0.0.1:PORT (也认小写 https_proxy)
: "${HTTPS_PROXY:=${https_proxy:-}}"

# 已验证的组合 (改这几个值请自己重新跑一遍精度/性能验证)
NCNN_REF="${NCNN_REF:-71f731859876c6da4195949a85a9c0b188304705}"   # 2026-11-17
HTTPLIB_REF="${HTTPLIB_REF:-v0.66.0}"
STB_REF="${STB_REF:-9600522de53973336fa033a289858a4354cbb04c}"      # stb_image 2.32 / write 1.18
STBIR_REF="${STBIR_REF:-a4f303e34a92d1410d510cbe85a1781ce4b8c9fd}"   # stb_image_resize2 v2.51

if [ ! -f third_party/ncnn/src/net.h ]; then
  echo "==> fetch ncnn @ ${NCNN_REF:0:12} (+ glslang submodule)"
  rm -rf third_party/ncnn
  git init -q third_party/ncnn
  git -C third_party/ncnn remote add origin https://github.com/Tencent/ncnn.git
  https_proxy="$HTTPS_PROXY" git -C third_party/ncnn fetch -q --depth 1 origin "$NCNN_REF"
  git -C third_party/ncnn checkout -q FETCH_HEAD
  ( cd third_party/ncnn
    https_proxy="$HTTPS_PROXY" git submodule update --init --depth 1 glslang )
else
  echo "==> ncnn already present"
fi

fetch() {  # $1=url $2=outfile $3=最小字节数
  local url="$1" out="third_party/include/$2" min="${3:-10000}"
  if [ -s "$out" ] && [ "$(stat -c%s "$out")" -ge "$min" ]; then
    echo "==> $2 present"
    return
  fi
  echo "==> fetch $2"
  rm -f "$out"
  https_proxy="$HTTPS_PROXY" curl -sSL --fail --max-time 300 "$url" -o "$out"
  local size
  size="$(stat -c%s "$out")"
  if [ "$size" -lt "$min" ]; then
    echo "!! $2 下载不完整 ($size B < $min B), 检查网络/代理或 URL" >&2
    rm -f "$out"
    exit 1
  fi
}

fetch "https://raw.githubusercontent.com/yhirose/cpp-httplib/$HTTPLIB_REF/httplib.h" \
      httplib.h 400000
fetch "https://raw.githubusercontent.com/nothings/stb/$STB_REF/stb_image.h" \
      stb_image.h 200000
fetch "https://raw.githubusercontent.com/nothings/stb/$STB_REF/stb_image_write.h" \
      stb_image_write.h 60000
fetch "https://raw.githubusercontent.com/nothings/stb_image_resize/$STBIR_REF/stb_image_resize2.h" \
      stb_image_resize2.h 500000

# 版本自检: 防止拿到错误页面 / 半截文件
grep -q "CPPHTTPLIB_VERSION" third_party/include/httplib.h || { echo "!! httplib.h 内容异常" >&2; exit 1; }
grep -q "stb_image - v" third_party/include/stb_image.h || { echo "!! stb_image.h 内容异常" >&2; exit 1; }
grep -q "stb_image_write - v" third_party/include/stb_image_write.h || { echo "!! stb_image_write.h 内容异常" >&2; exit 1; }
grep -q "stb_image_resize2 - v" third_party/include/stb_image_resize2.h || { echo "!! stb_image_resize2.h 内容异常" >&2; exit 1; }

echo "==> done  ($(grep -o 'CPPHTTPLIB_VERSION "[^"]*"' third_party/include/httplib.h | cut -d'"' -f2) / \
$(head -c 60 third_party/include/stb_image_resize2.h | grep -o 'v[0-9.]*' | head -1))"
