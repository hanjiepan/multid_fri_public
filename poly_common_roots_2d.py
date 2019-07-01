from __future__ import division
import numpy as np
import numexpr as ne
from scipy import linalg
import sympy


def find_roots_2d(coef1, coef2, tol=1e-3):
    """
    Find the common roots of two bivariate polynomials with coefficients specified by
    two 2D arrays.
    the variation along the first dimension (i.e., columns) is in the increasing order of y.
    the variation along the second dimension (i.e., rows) is in the increasing order of x.
    :param coef1: polynomial coefficients the first polynomial for the annihilation along rows
    :param coef2: polynomial coefficients the second polynomial for the annihilation along cols
    :return:
    """
    coef1 /= np.max(np.abs(coef1))
    coef2 /= np.max(np.abs(coef2))
    log_tol = np.log10(tol)
    # assert coef_col.shape[0] >= coef_row.shape[0] and coef_row.shape[1] >= coef_col.shape[1]
    if coef1.shape[1] < coef2.shape[1]:
        # swap input coefficients
        coef1, coef2 = coef2, coef1
    x, y = sympy.symbols('x, y')  # build symbols
    # collect both polynomials as a function of x; y will be included in the coefficients
    poly1 = 0
    poly2 = 0

    max_row_degree_y, max_row_degree_x = np.array(coef1.shape) - 1
    for x_count in range(max_row_degree_x + 1):
        for y_count in range(max_row_degree_y + 1):
            if np.abs(coef1[y_count, x_count]) > 1e-10:
                poly1 += coef1[y_count, x_count] * x ** (max_row_degree_x - x_count) * \
                         y ** (max_row_degree_y - y_count)
            else:
                coef1[y_count, x_count] = 0

    max_col_degree_y, max_col_degree_x = np.array(coef2.shape) - 1
    for x_count in range(max_col_degree_x + 1):
        for y_count in range(max_col_degree_y + 1):
            if np.abs(coef2[y_count, x_count]) > 1e-10:
                poly2 += coef2[y_count, x_count] * x ** (max_col_degree_x - x_count) * \
                         y ** (max_col_degree_y - y_count)
            else:
                coef2[y_count, x_count] = 0

    poly1_x = sympy.Poly(poly1, x)
    poly2_x = sympy.Poly(poly2, x)

    K_x = max_row_degree_x  # highest power of the first polynomial (in x)
    L_x = max_col_degree_x  # highest power of the second polynomial (in x)

    if coef1.shape[0] == 1:  # i.e., independent of variable y
        x_roots_all = np.roots(coef1.squeeze())
        eval_poly2 = sympy.lambdify(x, poly2)
        x_roots = []
        y_roots = []
        for x_loop in x_roots_all:
            y_roots_loop = np.roots(np.array(sympy.Poly(eval_poly2(x_loop), y).all_coeffs(), dtype=complex))
            y_roots.append(y_roots_loop)
            x_roots.append(np.tile(x_loop, y_roots_loop.size))
        coef_validate = coef2
    elif coef2.shape[1] == 1:  # i.e., independent of variable x
        y_roots_all = np.roots(coef2.squeeze())
        eval_poly1 = sympy.lambdify(y, poly1)
        x_roots = []
        y_roots = []
        for y_loop in y_roots_all:
            x_roots_loop = np.roots(np.array(sympy.Poly(eval_poly1(y_loop), x).all_coeffs(), dtype=complex))
            x_roots.append(x_roots_loop)
            y_roots.append(np.tile(y_loop, x_roots_loop.size))
        coef_validate = coef1
    else:
        if L_x >= 1:
            toep1_r = np.hstack((poly1_x.all_coeffs()[::-1], np.zeros(L_x - 1)))
            toep1_r = np.concatenate((toep1_r, np.zeros(L_x + K_x - toep1_r.size)))
            toep1_c = np.concatenate(([poly1_x.all_coeffs()[-1]], np.zeros(L_x - 1)))
        else:  # for the case with L_x == 0
            toep1_r = np.zeros((0, L_x + K_x))
            toep1_c = np.zeros((0, 0))

        if K_x >= 1:
            toep2_r = np.hstack((poly2_x.all_coeffs()[::-1], np.zeros(K_x - 1)))
            toep2_r = np.concatenate((toep2_r, np.zeros(L_x + K_x - toep2_r.size)))
            toep2_c = np.concatenate(([poly2_x.all_coeffs()[-1]], np.zeros(K_x - 1)))
        else:  # for the case with K_x == 0
            toep2_r = np.zeros((0, L_x + K_x))
            toep2_c = np.zeros((0, 0))

        blk_mtx1 = linalg.toeplitz(toep1_c, toep1_r)
        blk_mtx2 = linalg.toeplitz(toep2_c, toep2_r)
        if blk_mtx1.size != 0 and blk_mtx2.size != 0:
            mtx = np.vstack((blk_mtx1, blk_mtx2))
        elif blk_mtx1.size == 0 and blk_mtx2.size != 0:
            mtx = blk_mtx2
        elif blk_mtx1.size != 0 and blk_mtx2.size == 0:
            mtx = blk_mtx1
        else:
            mtx = np.zeros((0, 0))

        max_y_degree1 = coef1.shape[0] - 1

        max_y_degree2 = coef2.shape[0] - 1

        max_poly_degree = np.int(max_y_degree1 * L_x + max_y_degree2 * K_x)
        num_samples = (max_poly_degree + 1) * 8  # <= 4 is the over-sampling factor used to determined the poly coef.

        # randomly generate y-values
        # y_vals = np.random.randn(num_samples, 1) + \
        #          1j * np.random.randn(num_samples, 1)
        y_vals = np.exp(1j * 2 * np.pi / num_samples * np.arange(num_samples))[:, np.newaxis]
        y_powers = np.reshape(np.arange(max_poly_degree + 1)[::-1], (1, -1), order='F')
        Y = ne.evaluate('y_vals ** y_powers')

        # compute resultant, which is the determinant of mtx.
        # it is a polynomial in terms of variable y
        func_resultant = sympy.lambdify(y, sympy.Matrix(mtx))
        det_As = np.array([linalg.det(np.array(func_resultant(y_roots_loop), dtype=complex))
                           for y_roots_loop in y_vals.squeeze()], dtype=complex)
        coef_resultant = linalg.lstsq(Y, det_As)[0]

        # trim out very small coefficients
        # eps = np.max(np.abs(coef_resultant)) * tol
        # coef_resultant[np.abs(coef_resultant) < eps] = 0

        y_roots_all = np.roots(coef_resultant)
        # check if there're duplicated roots
        y_roots_all = eliminate_duplicate_roots(y_roots_all)

        # use the root values for y to find the root values for x
        # check if poly1_x or poly2_x are constant w.r.t. x
        if len(poly1_x.all_coeffs()) > 1:
            func_loop = sympy.lambdify(y, poly1_x.all_coeffs())
            coef_validate = coef2
        elif len(poly2_x.all_coeffs()) > 1:
            func_loop = sympy.lambdify(y, poly2_x.all_coeffs())
            coef_validate = coef1
        else:
            raise RuntimeError('Neither polynomials contain x')

        x_roots = []
        y_roots = []
        for loop in range(y_roots_all.size):
            y_roots_loop = y_roots_all[loop]
            x_roots_loop = np.roots(func_loop(y_roots_loop))
            # check if there're duplicated roots
            x_roots_loop = eliminate_duplicate_roots(x_roots_loop)
            for roots_loop in x_roots_loop:
                x_roots.append(roots_loop)
            for roots_loop in np.tile(y_roots_loop, x_roots_loop.size):
                y_roots.append(roots_loop)

    x_roots, y_roots = np.array(x_roots).flatten('F'), np.array(y_roots).flatten('F')
    x_roots, y_roots = eliminate_duplicate_roots_2d(x_roots, y_roots)
    # validate based on the polynomial values of the other polynomila
    # that is not used in the last step to get the roots
    poly_val = np.log10(np.abs(
        check_error_2d(coef_validate / linalg.norm(coef_validate.flatten()),
                       x_roots, y_roots)))

    # if the error is 2 orders larger than the smallest error, then we discard the root
    # print(poly_val)
    valid_idx = np.bitwise_or(poly_val < np.min(poly_val) + 2, poly_val < log_tol)
    x_roots = x_roots[valid_idx]
    y_roots = y_roots[valid_idx]

    '''
    Further verification with the resultant w.r.t. y, 
    which should also vanish at the common roots
    '''
    poly1_y = sympy.Poly(poly1, y)
    poly2_y = sympy.Poly(poly2, y)

    K_y = max_row_degree_y  # highest power of the first polynomial (in y)
    L_y = max_col_degree_y  # highest power of the second polynomial (in y)

    if L_y >= 1:
        toep1_r = np.hstack((poly1_y.all_coeffs()[::-1], np.zeros(L_y - 1)))
        toep1_r = np.concatenate((toep1_r, np.zeros(L_y + K_y - toep1_r.size)))
        toep1_c = np.concatenate(([poly1_y.all_coeffs()[-1]], np.zeros(L_y - 1)))
    else:  # for the case with L_y == 0
        toep1_r = np.zeros((0, L_y + K_y))
        toep1_c = np.zeros((0, 0))

    if K_y >= 1:
        toep2_r = np.hstack((poly2_y.all_coeffs()[::-1], np.zeros(K_y - 1)))
        toep2_r = np.concatenate((toep2_r, np.zeros(L_y + K_y - toep2_r.size)))
        toep2_c = np.concatenate(([poly2_y.all_coeffs()[-1]], np.zeros(K_y - 1)))
    else:  # for the case with K_y == 0
        toep2_r = np.zeros((0, L_y + K_y))
        toep2_c = np.zeros((0, 0))

    blk_mtx1 = linalg.toeplitz(toep1_c, toep1_r)
    blk_mtx2 = linalg.toeplitz(toep2_c, toep2_r)
    if blk_mtx1.size != 0 and blk_mtx2.size != 0:
        mtx = np.vstack((blk_mtx1, blk_mtx2))
    elif blk_mtx1.size == 0 and blk_mtx2.size != 0:
        mtx = blk_mtx2
    elif blk_mtx1.size != 0 and blk_mtx2.size == 0:
        mtx = blk_mtx1
    else:
        mtx = np.zeros((0, 0))

    func_resultant_verify = sympy.lambdify((x, y), sympy.Matrix(mtx))

    # evaluate the resultant w.r.t. y at the found roots. it should also vanish if
    # the pair is the common root
    res_y_val = np.zeros(x_roots.size, dtype=float)
    for loop in range(x_roots.size):
        res_y_val[loop] = \
            np.abs(linalg.det(
                np.array(
                    func_resultant_verify(x_roots[loop], y_roots[loop]),
                    dtype=complex
                )))

    log_res_y_val = np.log10(res_y_val)
    valid_idx = np.bitwise_or(log_res_y_val < np.min(log_res_y_val) + 2,
                              log_res_y_val < log_tol)
    x_roots = x_roots[valid_idx]
    y_roots = y_roots[valid_idx]
    return x_roots, y_roots


def check_error_2d(coef, x_val, y_val):
    val = 0
    max_degree_y, max_degree_x = np.array(coef.shape) - 1
    for x_count in range(max_degree_x + 1):
        for y_count in range(max_degree_y + 1):
            val += coef[y_count, x_count] * x_val ** (max_degree_x - x_count) * \
                   y_val ** (max_degree_y - y_count)
    return val


def eliminate_duplicate_roots(all_roots):
    total_roots = all_roots.size
    flags = np.ones(total_roots, dtype=bool)
    for loop_outer in range(total_roots - 1):
        root1 = all_roots[loop_outer]
        # compute the difference
        flags[loop_outer + 1 +
              np.where(np.abs(root1 - all_roots[loop_outer + 1:])
                       < 1e-2)[0]] = False

    return all_roots[flags]


def eliminate_duplicate_roots_2d(all_roots1, all_roots2):
    total_roots = all_roots1.size
    flags = np.ones(total_roots, dtype=bool)
    for loop_outer in range(total_roots - 1):
        root1 = all_roots1[loop_outer]
        root2 = all_roots2[loop_outer]
        # compute the difference
        flags[loop_outer + 1 +
              np.where(np.abs(
                  np.abs(root1 - all_roots1[loop_outer + 1:]) +
                  1j * np.abs(root2 - all_roots2[loop_outer + 1:])) < 1e-3
              )[0]] = False

    return all_roots1[flags], all_roots2[flags]


if __name__ == '__main__':
    coef_row = np.random.randn(3, 4)
    coef_col = np.random.randn(4, 3)
    x_roots, y_roots = find_roots_2d(coef_row, coef_col)
    # print x_roots, y_roots
    print(np.abs(check_error_2d(coef_row, x_roots, y_roots)))
