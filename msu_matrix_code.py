# code design by Lei Wang. 
# zhuifeng414@126.com
import numpy as np
import random
import time
from matplotlib import pyplot as plt, rcParams
rcParams['figure.dpi'] = 144

class bdm_class():
    def __init__(self):
        return

    def generate_random_list(self, d_dim):
        res = []
        for i in range(d_dim):
            item = random.random()
            if item == 0:
                item = 1e-10
            res.append(item)
        return res

    '''
    generate block-diagonal matrix
    '''
    def generate_bdm(self, n_dim, d_dim):
        return np.array([[self.generate_random_list(d_dim) for i in range(n_dim)] for j in range(n_dim)])

    '''
    unzip bdm matrix from n*n*d to n*n*d*d
    '''
    def unzip_bdm(self, x_bdm):
        (n_dim, n_dim, d_dim) = x_bdm.shape
        x_bdm_unzip = np.asarray([[np.diag(x_bdm[i][j])
            for j in range(n_dim)] for i in range(n_dim)])
        x_bdm_unzip = np.rollaxis(x_bdm_unzip, 2, 1)
        x_bdm_unzip = np.reshape(x_bdm_unzip, (n_dim * d_dim,) * 2)
        return x_bdm_unzip

    '''
    matrix add
    '''
    def add_bdm(self, x_bdm, y_bdm):
        # (n_dim_x, _, d_dim_x) = x_bdm.shape
        # (n_dim_y, _, d_dim_y) = y_bdm.shape
        # if n_dim_x == n_dim_y and d_dim_x == d_dim_y:
        #     return x_bdm + y_bdm
        # else:
        #     print('Error! Dimension Inconsistency!')
        #     return -1
        return x_bdm + y_bdm

    '''
    matrix transpose
    '''
    def transpose_bdm(self, x_bdm):
        return x_bdm.transpose(1,0,2)

    '''
    matrix multiply
    '''
    def multi_bdm(self, x_bdm, y_bdm):
        (n_dim_x, _, d_dim_x) = x_bdm.shape
        (n_dim_y, _, d_dim_y) = y_bdm.shape
        if n_dim_x == n_dim_y and d_dim_x == d_dim_y:
            return x_bdm*y_bdm
        else:
            print('Error! Dimension Inconsistency!')
            return -1

    '''
    inversion of diagonal matrix
    '''
    def fast_inverse_diag(self, x_array):
        return np.array([1/x for x in x_array])

    def get_divide_block(self, k_index, x_bdm):
        (n_dim_x, _, d_dim_x) = x_bdm.shape
        if not isinstance(k_index, int):
            print('Type of k_index error !')
        if k_index < 2 or k_index > n_dim_x:
            print('Range of K_index Error !')
        if k_index >= 2 and k_index <= n_dim_x:
            A11 = x_bdm[n_dim_x-k_index, n_dim_x-k_index].reshape(1, 1, d_dim_x)
            A12 = x_bdm[n_dim_x-k_index, n_dim_x-k_index+1:].reshape(1, k_index-1 , d_dim_x)
            A21 = x_bdm[n_dim_x-k_index+1:, n_dim_x-k_index].reshape(k_index-1, 1, d_dim_x)
            #A22 = x_bdm[n_dim_x-k_index+1:, n_dim_x-k_index+1:].reshape(k_index-1, k_index-1, d_dim_x)
            return A11, A12, A21


    def row_mul(self, matrix_a, matrix_b):
        dx_a, dy_a = matrix_a.shape
        dx_b, dy_b = matrix_b.shape
        row_mul_res = np.zeros(dy_a)
        if dx_a == dx_b and dy_a == dy_b:
            for i in range(dx_a):
                row_mul_res += matrix_a[i, :] * matrix_b[i, :]
            return row_mul_res
        else:
            print('dimension inconsistency!')


    def row_multi_block(self, matrix_a, matrix_b):
        _, dim_n, dim_d = matrix_b.shape
        res = []
        for i in range(dim_n):
            res.append(self.row_mul(matrix_a[0, :, :], matrix_b[:, i, :]))
        return np.array(res).reshape(1, dim_n, dim_d)


    def block_multi_col(self, matrix_a, matrix_b):
        dim_n, _, dim_d = matrix_a.shape
        res = []
        for i in range(dim_n):
            res.append(self.row_mul(matrix_a[i, :, :], matrix_b[:, 0, :]))
        return np.array(res).reshape(dim_n, 1, dim_d)


    def row_multi_col(self, matrix_a, matrix_b):
        _, dim_n, dim_d = matrix_a.shape
        res = np.zeros((dim_d))
        for i in range(dim_n):
            res += matrix_a[0, i, :] * matrix_b[i, 0, :]
        return np.array(res).reshape(1, 1, dim_d)


    def col_multi_row(self, matrix_a, matrix_b):
        dim_m, dim_n, dim_d = matrix_b.shape
        res = [[0 for i in range(dim_n)] for j in range(dim_n)]
        for i in range(dim_n):
            for j in range(dim_n):
                res[i][j] = self.row_multi_col(matrix_a[i, :, :].reshape(1, dim_m, dim_d),
                                          matrix_b[:, j, :].reshape(dim_m, 1, dim_d))
        return np.array(res).reshape(dim_n, dim_n, dim_d)


    def get_UL(self, A11, A12, inv_A22, A21):
        return self.fast_inverse_diag(A11 - self.row_multi_col(self.row_multi_block(A12, inv_A22), A21))


    def get_UR(self, F, A12, inv_A22):
        return self.row_multi_block(self.row_multi_block(-F, A12), inv_A22)


    def get_LL(self, inv_A22, A21, F):
        return self.block_multi_col(self.block_multi_col(-inv_A22, A21), F)


    def get_LR(self, inv_A22, A21, F, A12):
        return inv_A22 + self.col_multi_row(inv_A22, self.col_multi_row(self.col_multi_row(self.block_multi_col(A21, F), A12), inv_A22))

    def get_inv_M(self, inv_A22, A11, A12, A21):
        F = self.get_UL(A11, A12, inv_A22, A21)
        UR = self.get_UR(F, A12, inv_A22)
        LL = self.get_LL(inv_A22, A21, F)
        LR = self.get_LR(inv_A22, A21, F, A12)
        inv_M = np.concatenate((np.concatenate((F, UR), axis=1),
                        np.concatenate((LL, LR), axis=1)), axis=0)
        return inv_M

    '''
    inversion of block-diagonal matrix
    '''
    def inverse_bdm(self, x_bdm):
        _, dim_n, dim_d = x_bdm.shape
        H = [0 for i in range(dim_n+1)]
        H[1] = self.fast_inverse_diag(x_bdm[dim_n-1][dim_n-1]).reshape(1,1,dim_d)
        for i in range(2, dim_n+1):
            A11, A12, A21 = self.get_divide_block(i, x_bdm)
            H[i] = self.get_inv_M(H[i-1], A11, A12, A21)
        return H[-1]

    '''
    plot matrix
    '''
    def plot_matrix(self, x, **kwargs):
        vmax = max([np.max(_x) for _x in [x.real]])
        _x = getattr(x, 'real')
        mappable = plt.imshow(_x, cmap='coolwarm', vmax=vmax, vmin=-vmax)
        plt.colorbar(mappable)
        plt.show()
        return

    def get_time_cost(self, bdm, test_mode, dim_n_list, dim_d_list, seq_mode='dim_n'):
        zip_time_cost_res = []
        unzip_time_cost_res = []
        if seq_mode == 'dim_n':
            dim_d = int(dim_d_list[0])
            seq_list = dim_n_list
        elif seq_mode == 'dim_d':
            dim_n = int(dim_n_list[0])
            seq_list = dim_d_list

            #dim_n_list, dim_d_list = dim_d_list, dim_n_list
        for dim_i in seq_list:
            print(dim_i)
            test_number = 10
            if seq_mode == 'dim_n':
                dim_n = int(dim_i)
            elif seq_mode == 'dim_d':
                dim_d = int(dim_i)

            x_bdm = bdm.generate_bdm(dim_n, dim_d)
            y_bdm = bdm.generate_bdm(dim_n, dim_d)
            x_bdm_unzip = bdm.unzip_bdm(x_bdm)
            y_bdm_unzip = bdm.unzip_bdm(y_bdm)

            zip_time_cost = 0
            unzip_time_cost = 0
            for _ in range(test_number):
                start_zip_sub_time = time.time()
                if test_mode == 'add':
                    zip_res = bdm.add_bdm(x_bdm, y_bdm)
                elif test_mode == 'transpose':
                    zip_res = bdm.transpose_bdm(x_bdm)
                elif test_mode == 'multiply':
                    zip_res = bdm.multi_bdm(x_bdm, y_bdm)
                elif test_mode == 'inverse':
                    zip_res = bdm.inverse_bdm(x_bdm)
                end_zip_sub_time = time.time()
                zip_sub_time_cost = end_zip_sub_time - start_zip_sub_time
                zip_time_cost += zip_sub_time_cost

                start_unzip_sub_time = time.time()
                if test_mode == 'add':
                    unzip_res = x_bdm_unzip + y_bdm_unzip
                elif test_mode == 'transpose':
                    unzip_res = x_bdm_unzip.T
                elif test_mode == 'multiply':
                    unzip_res = x_bdm_unzip * y_bdm_unzip
                elif test_mode == 'inverse':
                    unzip_res = np.linalg.inv(x_bdm_unzip)
                end_unzip_sub_time = time.time()
                unzip_sub_time_cost = end_unzip_sub_time - start_unzip_sub_time
                unzip_time_cost += unzip_sub_time_cost
            zip_time_cost_res.append(zip_time_cost/test_number)
            unzip_time_cost_res.append(unzip_time_cost/test_number)
        return zip_time_cost_res, unzip_time_cost_res

def test_time_cost():
    bdm = bdm_class()
    dim_n_list = [1e1, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1]
    dim_d_list = [150]
    # dim_n_list = [3e1]
    # dim_d_list = [10, 50, 100, 150, 200, 250, 300, 350, 400]
    # test_mode_list = ['add', 'transpose', 'multiply']
    test_mode_list = ['inverse']
    color_list = ['r', 'g', 'b', 'y']
    seq_mode = 'dim_n'
    # test_mode = 'add'
    fig = plt.figure()
    for i in range(len(test_mode_list)):
        test_mode = test_mode_list[i]
        color_item = color_list[i]
        print(test_mode, color_item)
        zip_time_cost_res, unzip_time_cost_res = bdm.get_time_cost(bdm, test_mode, dim_n_list, dim_d_list, seq_mode)

        if seq_mode == 'dim_n':
            plt.plot(np.log(dim_n_list), zip_time_cost_res, '%s*-' % (color_item), label='zip-%s' % (test_mode))
            plt.plot(np.log(dim_n_list), unzip_time_cost_res, '%s.--' % (color_item),
                     label='unzip-%s' % (test_mode))
            plt.xlabel('log(dim_n)')
            plt.ylabel('time cost(s)')

        elif seq_mode == 'dim_d':
            plt.plot(dim_d_list, zip_time_cost_res, '%s*-' % (color_item), label='zip-%s' % (test_mode))
            plt.plot(dim_d_list, unzip_time_cost_res, '%s.--' % (color_item), label='unzip-%s' % (test_mode))
            plt.xlabel('dim_d')
            plt.ylabel('time cost(s)')

    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    dim_n = 5
    dim_d = 5
    bdm = bdm_class()
    x_bdm = bdm.generate_bdm(dim_n, dim_d)
    x_bdm_unzip = bdm.unzip_bdm(x_bdm)
    zip_res = bdm.inverse_bdm(x_bdm)
    unzip_res = np.linalg.inv(x_bdm_unzip)
    #bdm.plot_matrix(x_bdm_unzip)
    #bdm.plot_matrix(bdm.unzip_bdm(zip_res))
    #bdm.plot_matrix(unzip_res)
    #bdm.plot_matrix(bdm.unzip_bdm(zip_res) - unzip_res)
    #bdm.plot_matrix(bdm.unzip_bdm(zip_res).dot(x_bdm_unzip))
