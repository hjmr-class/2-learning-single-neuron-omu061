#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEACH_NUM (4)
#define INP_NUM (2)

const double alpha = 0.01;

double w[INP_NUM], dw[INP_NUM];
double theta, d_theta;

double teach_x[TEACH_NUM][INP_NUM] = {
    { 0, 0 },
    { 0, 1 },
    { 1, 0 },
    { 1, 1 }
};
double teach_y[TEACH_NUM] = { 0, 1, 1, 1 };

double rand_one()
{
    double r = random() / (double)RAND_MAX;
    return r;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double forward(double* x)
{
    int i = 0;
    double u = 0;
    for (i = 0; i < INP_NUM; i++) {
        u += w[i] * x[i];
    }
    u += theta;
    return sigmoid(u);
}

double func_error()
{
    int t = 0;
    double e = 0;
    for (t = 0; t < TEACH_NUM; t++) {
        double y = forward(teach_x[t]);
        e += 0.5 * (y - teach_y[t]) * (y - teach_y[t]);
    }
    return e;
}

void clear_dw()
{
    int i = 0;
    for (i = 0; i < INP_NUM; i++) {
        dw[i] = 0;
    }
    d_theta = 0;
}

void calc_dw(double* x_t, double y_hat)
{
    int i = 0;
    double y = forward(x_t);
    double dy = y * (1 - y);
    for (i = 0; i < INP_NUM; i++) {
        dw[i] += (y - y_hat) * dy * x_t[i];
    }
    d_theta += (y - y_hat) * dy;
}

void init_w()
{
    int i = 0;
    for (i = 0; i < INP_NUM + 1; i++) {
        w[i] = rand_one() * 2 - 1.0;
    }
    theta = rand_one() * 2 - 1.0;
}

void update_w()
{
    int i = 0;
    for (i = 0; i < INP_NUM; i++) {
        w[i] -= alpha * dw[i];
    }
    theta -= alpha * d_theta;
}

int main(void)
{
    int t = 0, loop = 0;

    init_w();
    for (loop = 0; loop < 100000; loop++) {

        if (loop % 1000 == 0) {
            printf("%d, %f\n", loop, func_error());
        }

        clear_dw();
        for (t = 0; t < TEACH_NUM; t++) {
            calc_dw(teach_x[t], teach_y[t]);
        }
        update_w();
    }
    printf("%d, %f\n", loop, func_error());

    for (t = 0; t < TEACH_NUM; t++) {
        double y = forward(teach_x[t]);
        printf("%d: y = %f <--> y_hat = %f\n", t, y, teach_y[t]);
    }
}