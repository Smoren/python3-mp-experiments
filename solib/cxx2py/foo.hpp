/* File: foo.hpp */

#include <iostream>

typedef int myint;

int foo(int a);  // supported

struct FooStruct {
  int a;
  int b;
};

class FooCls {
  FooCls(int a): a_(a) {}
  int get_a() { return a_; };
  private:
    int a_;
};

namespace ns {

  namespace ns2 {
  
    double bar(double x);  // supported
  }

  struct BarStruct {
    double a;
    double b;
  };

  class BarCls {
  public:
    BarCls(double a): a_(a) {}
    double get_a() { return a_; }
    static int fun() { return 54321; }  // supported
  private:
    double a_;
  };

}
