#pragma once

//int CUDAManager(std::vector<std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>> inSet);

double runPNSRcustom(std::shared_ptr<std::vector< cv::cuda::HostMem >> hostSrcArray,
    std::shared_ptr<std::vector< cv::cuda::HostMem >> hostDupeArray,
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuDupeArray,
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcBuf,
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuDupeBuf,
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuWriteBuf,
    std::shared_ptr< cv::cuda::GpuMat > gpuUnkBuf,
    std::shared_ptr<std::vector< double >> outArray,
    std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray);

int vectorBreaker(std::vector<std::vector<std::vector<cv::Mat, std::allocator<cv::Mat>>>> inSet);


int frameResize(std::shared_ptr<std::vector< cv::cuda::HostMem >> hostSrcArray,  //cpu mat on host memory
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuSrcArray,    //gpu mat on gpu memory
    std::shared_ptr<std::vector< cv::cuda::GpuMat >> gpuWriteBuf,    //gpu mat on gpu memory
    std::shared_ptr<std::vector< cv::cuda::Stream >> streamsArray);