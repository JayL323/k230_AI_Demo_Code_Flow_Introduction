import ulab.numpy as np                  #类似python numpy操作，但也会有一些接口不同
import nncase_runtime as nn              #nncase运行模块，封装了kpu（kmodel推理）和ai2d（图片预处理加速）操作
from media.camera import *               #摄像头模块
from media.display import *              #显示模块
from media.media import *                #软件抽象模块，主要封装媒体数据链路以及媒体缓冲区
import aidemo                            #aidemo模块，封装ai demo相关后处理、画图操作
import image                             #图像模块，主要用于读取、图像绘制元素（框、点等）等操作
import time                              #时间统计
import gc                                #垃圾回收模块
import os, sys                           #操作系统接口模块

#显示分辨率
DISPLAY_WIDTH = ALIGN_UP(1920, 16)
DISPLAY_HEIGHT = 1080

#AI分辨率
OUT_RGB888P_WIDTH = ALIGN_UP(224, 16)
OUT_RGB888P_HEIGH = 224

debug_mode=0

class ScopedTiming:
    def __init__(self, info="", enable_profile=True):
        self.info = info
        self.enable_profile = enable_profile

    def __enter__(self):
        if self.enable_profile:
            self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable_profile:
            elapsed_time = time.time_ns() - self.start_time
            print(f"{self.info} took {elapsed_time / 1000000:.2f} ms")

# 任务后处理
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

#********************for media_utils.py********************
global draw_img,osd_img                                     #for display
global buffer,media_source,media_sink                       #for media

# for display，已经封装好，无需自己再实现，直接调用即可，详细解析请查看1.6.2
def display_init():
    # hdmi显示初始化
    display.init(LT9611_1920X1080_30FPS)
    display.set_plane(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, PIXEL_FORMAT_YVU_PLANAR_420, DISPLAY_MIRROR_NONE, DISPLAY_CHN_VIDEO1)

def display_deinit():
    # 释放显示资源
    display.deinit()

def display_draw(label):
    # hdmi写文字
    with ScopedTiming("display_draw",debug_mode >0):
        global draw_img,osd_img

        if label:
            draw_img.clear()
            draw_img.draw_string(5,5,label,scale=5,color=(255,0,255,0))
            draw_img.copy_to(osd_img)
            display.show_image(osd_img, 0, 0, DISPLAY_CHN_OSD3)
        else:
            draw_img.clear()
            draw_img.copy_to(osd_img)
            display.show_image(osd_img, 0, 0, DISPLAY_CHN_OSD3)

#for camera
def camera_init(dev_id):
    # 设置摄像头类型
    camera.sensor_init(dev_id, CAM_DEFAULT_SENSOR)

     # （1）设置显示输出
    # 设置指定设备id的chn0的输出宽高
    camera.set_outsize(dev_id, CAM_CHN_ID_0, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    # 设置指定设备id的chn0的输出格式为yuv420sp
    camera.set_outfmt(dev_id, CAM_CHN_ID_0, PIXEL_FORMAT_YUV_SEMIPLANAR_420)

    # （2）设置AI输出
    # 设置指定设备id的chn2的输出宽高
    camera.set_outsize(dev_id, CAM_CHN_ID_2, OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH)
    # 设置指定设备id的chn2的输出格式为rgb88planar
    camera.set_outfmt(dev_id, CAM_CHN_ID_2, PIXEL_FORMAT_RGB_888_PLANAR)

def camera_start(dev_id):
    # camera启动
    camera.start_stream(dev_id)

def camera_read(dev_id):
    # 读取一帧图像
    with ScopedTiming("camera_read",debug_mode >0):
        rgb888p_img = camera.capture_image(dev_id, CAM_CHN_ID_2)
        return rgb888p_img

def camera_release_image(dev_id,rgb888p_img):
    # 释放一帧图像
    with ScopedTiming("camera_release_image",debug_mode >0):
        camera.release_image(dev_id, CAM_CHN_ID_2, rgb888p_img)

def camera_stop(dev_id):
    # 停止camera
    camera.stop_stream(dev_id)

#for media
def media_init():
    # meida初始化
    config = k_vb_config()
    config.max_pool_cnt = 1
    config.comm_pool[0].blk_size = 4 * DISPLAY_WIDTH * DISPLAY_HEIGHT
    config.comm_pool[0].blk_cnt = 1
    config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE

    media.buffer_config(config)

    global media_source, media_sink
    media_source = media_device(CAMERA_MOD_ID, CAM_DEV_ID_0, CAM_CHN_ID_0)
    media_sink = media_device(DISPLAY_MOD_ID, DISPLAY_DEV_ID, DISPLAY_CHN_VIDEO1)
    media.create_link(media_source, media_sink)

    # 初始化媒体buffer
    media.buffer_init()

    global buffer, draw_img, osd_img
    buffer = media.request_buffer(4 * DISPLAY_WIDTH * DISPLAY_HEIGHT)
    # 用于画框
    draw_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    # 用于拷贝画框结果，防止画框过程中发生buffer搬运
    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888, poolid=buffer.pool_id, alloc=image.ALLOC_VB,
                          phyaddr=buffer.phys_addr, virtaddr=buffer.virt_addr)

def media_deinit():
    # meida资源释放
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    if 'buffer' in globals():
        global buffer
        media.release_buffer(buffer)

    if 'media_source' in globals() and 'media_sink' in globals():
        global media_source, media_sink
        media.destroy_link(media_source, media_sink)

    media.buffer_deinit()

def classification():
    print("start")

    # 初始化参数
    kmodel_file = '/sdcard/k230_classify.kmodel'
    labels = ["bocai","changqiezi","huluobo","xihongshi","xilanhua"]
    confidence_threshold = 0.6
    num_classes= 5
    cls_idx=-1

    # 加载kmodel
    kpu = nn.kpu()
    kpu.load_kmodel(kmodel_file)

    # 摄像头初始化
    camera_init(CAM_DEV_ID_0)
    # 显示初始化
    display_init()

    try:
        media_init()
        # 启动摄像头
        camera_start(CAM_DEV_ID_0)

        while True:
            os.exitpoint()
            with ScopedTiming("total",debug_mode > 0):
                # 从摄像头拿取一帧数据
                rgb888p_img = camera_read(CAM_DEV_ID_0)
                # for rgb888planar
                if rgb888p_img.format() == image.RGBP888:
                    # rgb888（uint8,chw,rgb）->kmodel input（uint8,hwc,rgb）
                    # pre_process : chw -> hwc
                    ori_img_numpy = rgb888p_img.to_numpy_ref()
                    ori_img_copy = ori_img_numpy.copy()
                    shape=ori_img_copy.shape
                    img_tmp = ori_img_copy.reshape((shape[0], shape[1] * shape[2]))
                    img_tmp_trans = img_tmp.transpose()
                    img_res=img_tmp_trans.copy()
                    img_hwc=img_res.reshape((1,shape[1],shape[2],shape[0]))
                    input_tensor = nn.from_numpy(img_hwc)

                    # set kmodel input
                    kpu.set_input_tensor(0, input_tensor)

                    # kmodel run
                    kpu.run()

                    # get output
                    results = []
                    for i in range(kpu.outputs_size()):
                        output_tensor = kpu.get_output_tensor(i)
                        result = output_tensor.to_numpy()
                        del output_tensor
                        results.append(result)

                    # post process
                    softmax_res=softmax(results[0][0])
                    res_idx=np.argmax(softmax_res)
                    if softmax_res[res_idx]>confidence_threshold:
                        cls_idx=res_idx
                        print("classification result:")
                        print(labels[res_idx])
                        print("score",softmax_res[res_idx])
                    else:
                        cls_idx=-1

                # draw result
                if cls_idx>=0:
                    display_draw(labels[res_idx])
                else:
                    display_draw(None)

                # release image
                del input_tensor
                camera_release_image(CAM_DEV_ID_0,rgb888p_img)
                # release gc
                gc.collect()
                nn.shrink_memory_pool()
    except KeyboardInterrupt as e:
        print("user stop: ", e)
    except BaseException as e:
        sys.print_exception(e)
    finally:
        camera_stop(CAM_DEV_ID_0)
        display_deinit()
        del kpu
        gc.collect()
        nn.shrink_memory_pool()
        media_deinit()
    print("end")
    return 0


if __name__=="__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    nn.shrink_memory_pool()
    classification()
