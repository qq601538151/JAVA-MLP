package com.ai.enmu;

// 定义一个枚举类型 ActivationFunction，用于表示不同的激活函数
// 激活函数在神经网络中用于引入非线性特性，使网络能够学习复杂的模式
public enum ActivationFunction {
    // SIGMOID 是一种常用的激活函数，其输出范围在 0 到 1 之间
    SIGMOID {
        /**
         * 计算 SIGMOID 激活函数的值
         * @param x 输入值
         * @return SIGMOID 函数计算结果
         */
        @Override
        public double activate(double x) {
            // SIGMOID 函数公式：1 / (1 + e^(-x))
            // Math.exp(-x) 计算 e 的 -x 次幂
            return 1 / (1 + Math.exp(-x));
        }

        /**
         * 计算 SIGMOID 激活函数的导数
         * @param x 输入值
         * @return SIGMOID 函数导数的计算结果
         */
        @Override
        public double derivative(double x) {
            // 先计算 SIGMOID 函数值
            double sig = activate(x);
            // SIGMOID 函数导数公式：sig * (1 - sig)
            return sig * (1 - sig);
        }
    },
    // RELU 是一种常用的激活函数，当输入大于 0 时，输出等于输入；当输入小于等于 0 时，输出为 0
    RELU {
        /**
         * 计算 RELU 激活函数的值
         * @param x 输入值
         * @return RELU 函数计算结果
         */
        @Override
        public double activate(double x) {
            // RELU 函数：max(0, x)
            // Math.max(0, x) 返回 0 和 x 中的较大值
            return Math.max(0, x);
        }

        /**
         * 计算 RELU 激活函数的导数
         * @param x 输入值
         * @return RELU 函数导数的计算结果
         */
        @Override
        public double derivative(double x) {
            // 当 x 大于 0 时导数为 1，否则为 0
            return x > 0 ? 1 : 0;
        }
    },
    // TANH 是一种常用的激活函数，其输出范围在 -1 到 1 之间
    TANH {
        /**
         * 计算 TANH 激活函数的值
         * @param x 输入值
         * @return TANH 函数计算结果
         */
        @Override
        public double activate(double x) {
            // TANH 函数
            // Math.tanh(x) 计算双曲正切值
            return Math.tanh(x);
        }

        /**
         * 计算 TANH 激活函数的导数
         * @param x 输入值
         * @return TANH 函数导数的计算结果
         */
        @Override
        public double derivative(double x) {
            // TANH 函数导数公式：1 - tanh(x)^2
            return 1 - Math.pow(Math.tanh(x), 2);
        }
    };

    // 抽象方法，用于计算激活函数值
    public abstract double activate(double x);

    // 抽象方法，用于计算激活函数的导数
    public abstract double derivative(double x);
}