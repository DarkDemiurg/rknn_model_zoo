#ifndef __QUEUE_H__
#define __QUEUE_H__
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>

using namespace std;

// simple buffer struct for V4L2 mmap plane0 mapping used by this driver
struct buffer {
    void* start = nullptr;
    size_t length = 0;
    size_t capacity = 0;
};

// Generic DMA buffer wrapper (can allocate any byte size)
struct DMABuffer {
    void* va = nullptr;   // virtual addr
    int   fd = -1;        // dma-buf fd
    size_t size = 0;

    DMABuffer() = default;

    bool alloc(const char* heap, size_t bytes) {
        // free if allocated
        if (va && fd >= 0) {
            dma_buf_free(size, &fd, va);
            va = nullptr; fd = -1; size = 0;
        }
        int ret = dma_buf_alloc(heap, bytes, &fd, &va);
        if (ret < 0) {
            va = nullptr; fd = -1; size = 0;
            return false;
        }
        size = bytes;
        return true;
    }

    void free_buf() {
        std::cout << "free DMA buffer" << std::endl;
        if (va && fd >= 0) {
            dma_buf_free(size, &fd, va);
            va = nullptr; fd = -1; size = 0;
        }
    }

    ~DMABuffer() { free_buf(); }
};

// simple threadsafe queue (non-blocking pop)
template<typename T>
class TSQueue {
    std::queue<T> q;
    std::mutex m;
    std::condition_variable cv;
    size_t maxsz;
public:
    TSQueue(size_t maxsize = 32): maxsz(maxsize) {}
    bool push(T v) {
        std::lock_guard<std::mutex> lk(m);
        if (q.size() >= maxsz) return false;
        q.push(std::move(v));
        cv.notify_one();
        return true;
    }
    bool pop(T &out) {
        std::lock_guard<std::mutex> lk(m);
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop();
        return true;
    }
    void wait_pop(T &out) {
       std:: unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]{ return !q.empty(); });
        out = std::move(q.front()); q.pop();
    }
};

// NV12 frame stored in a DMABuffer (size = w * h * 3 / 2)
struct NV12FrameDMABuf {
    shared_ptr<DMABuffer> buf;
    int width;
    int height;
    size_t size() const { return buf ? buf->size : 0; }
};

// MemoryPool: preallocate NV12 DMABuffers and manage idle/data queues
class MemoryPool {
    vector<shared_ptr<TSQueue<shared_ptr<NV12FrameDMABuf>>>> idleQueues;
    vector<shared_ptr<TSQueue<shared_ptr<NV12FrameDMABuf>>>> dataQueues;
public:
    MemoryPool(size_t pool_size_per_cam, int cam_count, int width, int height) : width_(width), height_(height) {
        for (int i = 0; i < cam_count; ++i) {
            auto idle = make_shared<TSQueue<shared_ptr<NV12FrameDMABuf>>>(pool_size_per_cam + 2); // +2 is order to avoid deadlock
            auto data = make_shared<TSQueue<shared_ptr<NV12FrameDMABuf>>>(pool_size_per_cam + 2);
            // prealloc pool_size_per_cam DMA NV12 buffers
            for (size_t j = 0; j < pool_size_per_cam; ++j) {
                auto frame = make_shared<NV12FrameDMABuf>();
                frame->width = width_;
                frame->height = height_;
                frame->buf = make_shared<DMABuffer>();
                size_t nv12_size = size_t(width_) * height_ * 3 / 2;
                if (!frame->buf->alloc(DMA_HEAP_DMA32_UNCACHED_PATH, nv12_size)) {
                    cerr << "dma alloc failed for NV12 buffer\n";
                    throw runtime_error("dma alloc failed");
                }
                idle->push(frame);
            }
            idleQueues.push_back(idle);
            dataQueues.push_back(data);
        }
    }

    shared_ptr<TSQueue<shared_ptr<NV12FrameDMABuf>>> getIdleQueue(int cam_id) {
        if (cam_id < 0 || cam_id >= (int)idleQueues.size()) return nullptr;
        return idleQueues[cam_id];
    }
    shared_ptr<TSQueue<shared_ptr<NV12FrameDMABuf>>> getDataQueue(int cam_id) {
        if (cam_id < 0 || cam_id >= (int)dataQueues.size()) return nullptr;
        return dataQueues[cam_id];
    }
private:
    int width_;
    int height_;
};
#endif
