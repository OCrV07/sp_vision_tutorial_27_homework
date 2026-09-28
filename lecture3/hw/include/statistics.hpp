#pragma once

#include <mutex>

struct StatisticsSnapshot
{
    int produced = 0;
    int processed = 0;
    int saved = 0;
    int corrupted = 0;
};

class Statistics
{
public:
    void onProduced();
    void onProcessed();
    void onSaved();
    void onCorrupted();
    StatisticsSnapshot snapshot() const;

private:
    // 多个线程同时执行“读取旧值、加一、写回”的操作，可能发生数据竞争，导致计数丢失
    // 引入 <mutex>
    // 在各个函数中加锁, 保证每次计数更新都是完整、不可被其他线程打断的
    // 并确保函数退出时自动释放锁，避免异常或提前返回造成死锁
    mutable std::mutex mutex_;
    int produced_ = 0;
    int processed_ = 0;
    int saved_ = 0;
    int corrupted_ = 0;
};
