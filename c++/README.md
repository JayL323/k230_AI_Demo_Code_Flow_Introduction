**执行步骤**：（本执行步骤需要对K230有一定程度了解，若是刚接触，则只需关注代码实现部分）

1. 下载镜像：[k230_canmv_sdcard_v1.3_nncase_v2.7.0.img.gz](https://kendryte-download.canaan-creative.com/developer/k230/k230_canmv_sdcard_v1.3_nncase_v2.7.0.img.gz)
2. 将镜像烧录到K230-CanMV开发板，并启动
3. 将k230_bin目录拷贝到开发板/sharefs
4. 在大核中，进入/sharefs/k230_bin目录，执行k230_classify_isp.sh
