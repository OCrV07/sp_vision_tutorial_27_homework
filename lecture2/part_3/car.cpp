#include <iostream>
using namespace std;

class Car {
    public :
        Car();
        ~Car();
        void run();
    
    private :
        int count_;
    
};

Car::Car() {
    cout << "Car 构造完成" << endl;
    count_ = 0;
}

Car::~Car() {
    cout << "Car 析构完成" << endl;
}

int main() {
    Car car;
    car.run();
}