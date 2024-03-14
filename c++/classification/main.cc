/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <thread>
#include <map>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "vi_vo.h"

using namespace nncase;
using namespace nncase::runtime;

using cv::Mat;
using std::cerr;
using std::cout;
using std::endl;
using namespace std;

auto cache = cv::Mat::zeros(1, 1, CV_32FC1);
/**
 * @brief 分类结果结构
 */
typedef struct cls_res
{
    float score;//分类分数
    string label;//分类标签结果
}cls_res;

std::atomic<bool> isp_stop(false);

void video_proc_cls(char *argv[])
{
    
    /***************************fixed：无需修改***********************************/
    vivcap_start();

    // osd set
    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory for sensor
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
    /*****************************************************************************/ 

    string kmodel_path = argv[1];
    cout<<"kmodel_path : "<<kmodel_path<<endl;
    float cls_thresh=0.5;

    /***************************fixed：无需修改***********************************/
    // 我们已经把相关实现封装到ai_base.cc，这里只是为了介绍起来比较简单
    interpreter kmodel_interp;        
    // load model
    std::ifstream ifs(kmodel_path, std::ios::binary);
    kmodel_interp.load_model(ifs).expect("Invalid kmodel");

    // inputs init
    for (size_t i = 0; i < kmodel_interp.inputs_size(); i++)
    {
        auto desc = kmodel_interp.input_desc(i);
        auto shape = kmodel_interp.input_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create input tensor");
        kmodel_interp.input_tensor(i, tensor).expect("cannot set input tensor");
    } 
    auto shape0 = kmodel_interp.input_shape(0);      //nhwc
    int kmodel_input_height = shape0[1];
    int kmodel_input_width = shape0[2];

    // outputs init
    for (size_t i = 0; i < kmodel_interp.outputs_size(); i++)
    {
        auto desc = kmodel_interp.output_desc(i);
        auto shape = kmodel_interp.output_shape(i);
        auto tensor = host_runtime_tensor::create(desc.datatype, shape, hrt::pool_shared).expect("cannot create output tensor");
        kmodel_interp.output_tensor(i, tensor).expect("cannot set output tensor");
    }
    /*****************************************************************************/ 

    vector<cls_res> results;
    std::vector<std::string> labels = {"bocai","changqiezi","huluobo","xihongshi","xilanhua"};
    
    while (!isp_stop)
    {
        cv::Mat ori_img;
        //sensor to cv::Mat
        {
            /***************************fixed：无需修改***********************************/
            //从摄像头读取一帧图像
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }

            //将摄像头当前帧对应DDR地址映射到当前系统进行访问
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3); 
            kd_mpi_sys_munmap(vbvaddr, size);
            /*****************************************************************************/ 
            
            //将摄像头数据转换为为cv::Mat,sensor（rgb,chw）->cv::Mat（bgr，hwc）
            cv::Mat image_r = cv::Mat(SENSOR_HEIGHT,SENSOR_WIDTH, CV_8UC1, vaddr);
            cv::Mat image_g = cv::Mat(SENSOR_HEIGHT,SENSOR_WIDTH, CV_8UC1, vaddr+SENSOR_HEIGHT*SENSOR_WIDTH);
            cv::Mat image_b = cv::Mat(SENSOR_HEIGHT,SENSOR_WIDTH, CV_8UC1, vaddr+2*SENSOR_HEIGHT*SENSOR_WIDTH);
            std::vector<cv::Mat> color_vec(3);
            color_vec.clear();
            color_vec.push_back(image_b);
            color_vec.push_back(image_g);
            color_vec.push_back(image_r);
            cv::merge(color_vec, ori_img);
        }

        /***************************unfixed：不同AI Demo可能需要修改******************/
        // pre_process
        cv::Mat pre_process_img;
        {
            cv::Mat rgb_img;
            cv::cvtColor(ori_img, rgb_img, cv::COLOR_BGR2RGB);
            cv::resize(rgb_img, pre_process_img, cv::Size(kmodel_input_width, kmodel_input_height), cv::INTER_LINEAR);
        }
        /*****************************************************************************/  

        /***************************fixed：无需修改***********************************/
        // set kmodel input
        {
            runtime_tensor tensor0 = kmodel_interp.input_tensor(0).expect("cannot get input tensor");
            auto in_buf = tensor0.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
            memcpy(reinterpret_cast<unsigned char *>(in_buf.data()), pre_process_img.data,sizeof(uint8_t)* kmodel_input_height * kmodel_input_width * 3);
            hrt::sync(tensor0, sync_op_t::sync_write_back, true).expect("sync write_back failed");
        }
        

        // kmodel run
        kmodel_interp.run().expect("error occurred in running model");

        // get kmodel output
        vector<float *> k_outputs;
        {
            for (int i = 0; i < kmodel_interp.outputs_size(); i++)
            {
                auto out = kmodel_interp.output_tensor(i).expect("cannot get output tensor");
                auto buf = out.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_read).unwrap().buffer();
                float *p_out = reinterpret_cast<float *>(buf.data());
                k_outputs.push_back(p_out);
            }
        }
        /***************************fixed：无需修改***********************************/

        /***************************unfixed：不同AI Demo可能需要修改******************/
        //post process
        results.clear();
        {
            
            float* output0 = k_outputs[0];
            float sum = 0.0;
            for (int i = 0; i < labels.size(); i++){
                sum += exp(output0[i]);
            }
            
            int max_index;
            for (int i = 0; i < labels.size(); i++)
            {
                output0[i] = exp(output0[i]) / sum;
            }
            max_index = std::max_element(output0,output0+labels.size()) - output0; 
            cls_res b;
            if (output0[max_index] >= cls_thresh)
            {
                b.label = labels[max_index];
                b.score = output0[max_index];
                results.push_back(b);
            }
        }
        /*****************************************************************************/    

        // draw result to vo
        {
            {
                cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
                {
                    /***************************unfixed：不同AI Demo可能需要修改******************/
                    //draw cls
                    double fontsize = (osd_frame.cols * osd_frame.rows * 1.0) / (1100 * 1200);
                    for(int i = 0; i < results.size(); i++)
                    {   
                        std::string text = "class: " + results[i].label + ", score: " + std::to_string(round(results[i].score * 100) / 100.0).substr(0, 4);

                        cv::putText(osd_frame, text, cv::Point(1, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255, 0), 2);

                        std::cout << text << std::endl;
                    }
                    /*****************************************************************************/
                }

                /***************************fixed：无需修改***********************************/
                memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            }

            // insert osd to vo
            {
                kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
                printf("kd_mpi_vo_chn_insert_frame success \n");
            }
            
        }
        
        {
            // 释放sensor当前帧
            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
        /*****************************************************************************/ 
    }
    
    /***************************fixed：无需修改***********************************/
    vo_osd_release_block();
    vivcap_stop();

    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
    /*****************************************************************************/ 
}

int main(int argc, char *argv[])
{
     std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 2)
    {
        cout << "模型推理时传参说明："
         << "<kmodel_path>" << endl
         << "Options:" << endl
         << "  kmodel_path     Kmodel的路径\n"
         << "\n"
         << endl;
        return -1;
    }

    /***************************fixed：无需修改***********************************/
    std::thread thread_isp(video_proc_cls, argv);
    while (getchar() != 'q')
    {
        usleep(10000);
    }

    isp_stop = true;
    thread_isp.join();
    /*****************************************************************************/ 
    return 0;
}