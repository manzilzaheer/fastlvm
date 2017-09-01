#ifndef _TASKQUEUE_H
#define _TASKQUEUE_H

#include <atomic>

class task_queue
{
    std::atomic<unsigned> pos;
    unsigned _size;

public:
    task_queue(unsigned len=0) : pos{0}, _size(len)
    {   }

    inline unsigned pop()
    {
        unsigned top = pos.fetch_add(1, std::memory_order_relaxed);
        if (top < _size)
            top += 1;
        else
            top = 0;
        return top;
    }

    inline void reset()
    {
        pos.store(0, std::memory_order_relaxed);
    }

    inline void resize(unsigned len)
    {
        _size = len;
    }
};

#endif