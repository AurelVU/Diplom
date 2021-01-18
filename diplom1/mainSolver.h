#pragma once
#include <string>
#include <stdio.h>

using namespace std;

class mainSolver
{
protected:
    double FaceArea;
    double k;
    double delt;
public:
    double** T;
    int imax;
    int jmax;
    double dh;

    bool create(string filename_);

    void RunPhysic();

    //Insulated
    void RunPhysic2();
};