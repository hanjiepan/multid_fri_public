from __future__ import division
import numpy as np
import numexpr as ne
from scipy import linalg
import sympy
from poly_common_roots_2d import check_error_2d


def find_roots_3d(coef1, coef2, coef3, tol=1e-3):
    """
    Find the common roots of the two polynomials with coefficients specified
    by three 3D arrays.
    the variation along the first dimension (i.e., columns) is in the increasing order of y.
    the variation along the second dimension (i.e., rows) is in the increasing order of x.
    the variation along the third dimension (i.e., depth) is in the increasing order of z.
    :param coef1: polynomial coefficients the first polynomial for the annihilation along ROWS
    :param coef2: polynomial coefficients the second polynomial for the annihilation along COLUMNS
    :param coef3: polynomial coefficients the third polynomial for the annihilation along DEPTH
    :return:
    """
    coef1 /= np.max(np.abs(coef1))
    coef2 /= np.max(np.abs(coef2))
    coef3 /= np.max(np.abs(coef3))
    assert coef1.shape[1] >= coef2.shape[1] and coef1.shape[1] >= coef3.shape[1]
    assert coef2.shape[0] >= coef1.shape[0] and coef2.shape[0] >= coef3.shape[0]
    assert coef3.shape[2] >= coef1.shape[2] and coef3.shape[2] >= coef2.shape[2]
    x, y, z = sympy.symbols('x, y, z')  # build symbols

    # convert coefficient to polynomials
    poly1 = coef2poly_3d(coef1, x, y, z)
    poly2 = coef2poly_3d(coef2, x, y, z)
    poly3 = coef2poly_3d(coef3, x, y, z)

    # collete them with respect to x: the coefficients are expressions of y and z
    poly1_x = sympy.Poly(poly1, x)
    poly2_x = sympy.Poly(poly2, x)
    poly3_x = sympy.Poly(poly3, x)

    if coef1.shape[0] == 1 and coef1.shape[2] == 1:  # i.e., independent of variable y, z
        x_roots_all = np.roots(coef1.squeeze())
        # for each x we use the 2D root finding routine
        x_roots = []
        y_roots = []
        z_roots = []
        for x_root_loop in x_roots_all:
            poly2_coef_yz = compute_coef_yz(x_root_loop, coef2)
            poly3_coef_yz = compute_coef_yz(x_root_loop, coef3)
            z_roots_loop, y_roots_loop = find_roots_2d(poly3_coef_yz, poly2_coef_yz)
            for root_loop in np.tile(x_root_loop, z_roots_loop.size):
                x_roots.append(root_loop)
            for root_loop in y_roots_loop:
                y_roots.append(root_loop)
            for root_loop in z_roots_loop:
                z_roots.append(root_loop)
    elif coef2.shape[1] == 1 and coef2.shape[2] == 1:  # i.e., independent of x, z
        y_roots_all = np.roots(coef2.squeeze())
        # for each y we use the 2D root finding routine
        x_roots = []
        y_roots = []
        z_roots = []
        for y_root_loop in y_roots_all:
            poly1_coef_xz = compute_coef_xz(y_root_loop, coef1)
            poly3_coef_xz = compute_coef_xz(y_root_loop, coef3)
            z_roots_loop, x_roots_loop = find_roots_2d(poly3_coef_xz, poly1_coef_xz)
            for root_loop in np.tile(y_root_loop, z_roots_loop.size):
                y_roots.append(root_loop)
            for root_loop in x_roots_loop:
                x_roots.append(root_loop)
            for root_loop in z_roots_loop:
                z_roots.append(root_loop)
    elif coef3.shape[0] == 1 and coef3.shape[1] == 1:  # i.e., independent of x, y
        z_roots_all = np.roots(coef3.squeeze())
        # for each z we use the 2D root finding routine
        x_roots = []
        y_roots = []
        z_roots = []
        for z_root_loop in z_roots_all:
            poly1_coef_xy = compute_coef_xy(z_root_loop, coef1)
            poly2_coef_xy = compute_coef_xy(z_root_loop, coef2)
            x_roots_loop, y_roots_loop = find_roots_2d(poly1_coef_xy, poly2_coef_xy)
            for root_loop in np.tile(z_root_loop, x_roots_loop.size):
                z_roots.append(root_loop)
            for root_loop in x_roots_loop:
                x_roots.append(root_loop)
            for root_loop in y_roots_loop:
                y_roots.append(root_loop)
    else:  # the general case
        # first compute the resultant between filter 1 (horizontal direction dominating)
        # and filter 2 (vertical direction dominating)
        coef_resultant_1_2 = compute_resultant_3d(poly1_x, poly2_x, y, z)
        coef_resultant_2_3 = compute_resultant_3d(poly2_x, poly3_x, y, z)
        z_roots_all, y_roots_all = find_roots_2d(coef_resultant_1_2, coef_resultant_2_3)
        z_roots_all = z_roots_all.flatten('F')
        y_roots_all = y_roots_all.flatten('F')

        # # use the resultant between filter 1 and 2 as verification
        # coef_resultant_1_2 = compute_resultant_3d(poly1_x, poly2_x, y, z, tol=tol)
        # poly_val_resultant_veri = np.log10(np.abs(
        #     check_error_2d(coef_resultant_1_2 / linalg.norm(coef_resultant_1_2.flatten()),
        #                    z_roots_all, y_roots_all)))
        # # if the error is 2 orders larger than the smallest error, then we discard the root
        # # print(poly_val)
        # valid_idx = np.bitwise_or(poly_val_resultant_veri < np.min(poly_val_resultant_veri) + 2,
        #                           poly_val_resultant_veri < log_tol)
        # z_roots_all = z_roots_all[valid_idx]
        # y_roots_all = y_roots_all[valid_idx]

        # take the z_roots, and y_roots to filter 1 and get the roots of x
        x_roots = []
        y_roots = []
        z_roots = []
        func_poly1_yz = sympy.lambdify((y, z), poly1_x.all_coeffs())
        for y_root_loop, z_root_loop in zip(y_roots_all, z_roots_all):
            poly_coef_x_loop = np.array(func_poly1_yz(y_root_loop, z_root_loop))
            x_root_loop = np.roots(poly_coef_x_loop)
            for root_loop in x_root_loop:
                x_roots.append(root_loop)
            for root_loop in np.tile(y_root_loop, x_root_loop.size):
                y_roots.append(root_loop)
            for root_loop in np.tile(z_root_loop, x_root_loop.size):
                z_roots.append(root_loop)

    x_roots, y_roots, z_roots = \
        np.array(x_roots).flatten('F'), \
        np.array(y_roots).flatten('F'), \
        np.array(z_roots).flatten('F')
    # x_roots, y_roots, z_roots = eliminate_duplicate_roots_3d(x_roots, y_roots, z_roots)
    # evaluate three polynomials and eliminate spurious roots
    poly1_vals = np.abs(check_error_3d(coef1 / linalg.norm(coef1.flatten()),
                                       x_roots, y_roots, z_roots))
    poly2_vals = np.abs(check_error_3d(coef2 / linalg.norm(coef2.flatten()),
                                       x_roots, y_roots, z_roots))
    poly3_vals = np.abs(check_error_3d(coef3 / linalg.norm(coef3.flatten()),
                                       x_roots, y_roots, z_roots))
    valid_idx = np.bitwise_and(poly1_vals < tol,
                               poly2_vals < tol,
                               poly3_vals < tol)
    x_roots = np.atleast_1d(x_roots[valid_idx].squeeze())
    y_roots = np.atleast_1d(y_roots[valid_idx].squeeze())
    z_roots = np.atleast_1d(z_roots[valid_idx].squeeze())

    # # TODO: remove after debugging
    # data_gt = np.load('./result/gt_b.npz')
    # uk_gt = np.exp(-2j * np.pi * data_gt['xk'])
    # vk_gt = np.exp(-2j * np.pi * data_gt['yk'])
    # wk_gt = np.exp(-2j * np.pi * data_gt['zk'])
    # print(np.abs(check_error_3d(coef1, uk_gt, vk_gt, wk_gt)))
    # print(np.abs(check_error_3d(coef2, uk_gt, vk_gt, wk_gt)))
    # print(np.abs(check_error_3d(coef3, uk_gt, vk_gt, wk_gt)))

    return x_roots, y_roots, z_roots


def compute_resultant_3d(poly1_x, poly2_x, y, z, tol=1e-10):
    """
    compute resultant of two polynomials (collected with respect to x.
    Dimension 0 of the resultant coefficient corresponds to the power of y.
    Dimension 1 of the resultant coefficient corresponds to the power of z.
    The highest power term for both y and z corresponds to the upper left corner.
    :param poly1_x: a symoblic polynomial (defined as sympy.Poly(., x)
    :param poly2_x: a symoblic polynomial (defined as sympy.Poly(., x)
    :param y: the symbol for the vertical direction
    :param z: the symbol for the horizontal direction
    :return:
    """
    K_x = len(poly1_x.all_coeffs()) - 1
    L_x = len(poly2_x.all_coeffs()) - 1
    if L_x >= 1:
        toep1_r = np.hstack((poly1_x.all_coeffs()[::-1], np.zeros(L_x - 1)))
        toep1_c = np.concatenate(([poly1_x.all_coeffs()[-1]], np.zeros(L_x - 1)))
    else:
        toep1_r = np.zeros((0, L_x + K_x))
        toep1_c = np.zeros((0, 0))

    if K_x >= 1:
        toep2_r = np.hstack((poly2_x.all_coeffs()[::-1], np.zeros(K_x - 1)))
        toep2_c = np.concatenate(([poly2_x.all_coeffs()[-1]], np.zeros(K_x - 1)))
    else:
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

    max_y_degree1 = len(sympy.Poly(poly1_x, y).all_coeffs()) - 1
    max_z_degree1 = len(sympy.Poly(poly1_x, z).all_coeffs()) - 1
    max_y_degree2 = len(sympy.Poly(poly2_x, y).all_coeffs()) - 1
    max_z_degree2 = len(sympy.Poly(poly2_x, z).all_coeffs()) - 1

    max_poly_degree_y = np.int(max_y_degree1 * L_x + max_y_degree2 * K_x)
    max_poly_degree_z = np.int(max_z_degree1 * L_x + max_z_degree2 * K_x)
    # 4 is the over-sampling factor used to determined the poly coef.
    num_samples_y = (max_poly_degree_y + 1) * 2
    num_samples_z = (max_poly_degree_z + 1) * 2
    y_vals = np.exp(1j * 2 * np.pi / num_samples_y * np.arange(num_samples_y))[:, np.newaxis]
    z_vals = np.exp(1j * 2 * np.pi / num_samples_z * np.arange(num_samples_z))[:, np.newaxis]
    z_vals_mesh, y_vals_mesh = np.meshgrid(z_vals, y_vals)
    # y_vals_mesh = np.exp(1j * 2 * np.pi * np.random.rand(num_samples_y * num_samples_z))
    # z_vals_mesh = np.exp(1j * 2 * np.pi * np.random.rand(num_samples_y * num_samples_z))
    y_vals_mesh = np.reshape(y_vals_mesh, (-1, 1), order='F')
    z_vals_mesh = np.reshape(z_vals_mesh, (-1, 1), order='F')
    z_powers, y_powers = np.meshgrid(np.arange(max_poly_degree_z + 1)[::-1],
                                     np.arange(max_poly_degree_y + 1)[::-1])
    z_powers = np.reshape(z_powers, (1, -1), order='F')
    y_powers = np.reshape(y_powers, (1, -1), order='F')
    YZ = ne.evaluate('y_vals_mesh ** y_powers * z_vals_mesh ** z_powers')

    func_resultant = sympy.lambdify((y, z), sympy.Matrix(mtx))
    det_As = np.array([linalg.det(np.array(func_resultant(y_val_loop, z_val_loop), dtype=complex))
                       for y_val_loop, z_val_loop in zip(y_vals_mesh.squeeze(), z_vals_mesh.squeeze())], dtype=complex)
    coef_resultant = linalg.lstsq(YZ, det_As)[0]

    # trim out very small coefficients
    # eps = np.max(np.abs(coef_resultant)) * tol
    # coef_resultant[np.abs(coef_resultant) < eps] = 0
    coef_resultant = np.reshape(coef_resultant, (max_poly_degree_y + 1, -1), order='F')

    return coef_resultant


def poly2coef_2d(expression, symbol_h, symbol_v, poly_degree_h, poly_degree_v):
    """
    extract the polynomial coefficients and put them in a 2D block
    :param expression: polynomial expression in terms of variable symbol_h and symbol_v
    :param symbol_h: symbol used for the horizontal direction
    :param symbol_v: symbol used for the vertical direction
    :param poly_degree_h: maximum degree of the horizontal direction
    :param poly_degree_v: maximum degree of the vertical direction
    :return:
    """
    coef_blk = np.zeros((poly_degree_v + 1, poly_degree_h + 1))
    # get polynomial coefficients w.r.t. symbol_h
    coef_h = sympy.Poly(expression, symbol_h).all_coeffs()
    # fillin coef_blk column by column
    col_count = -1
    for coef_h_loop in coef_h[::-1]:
        coef_v_loop = sympy.Poly(coef_h_loop, symbol_v).all_coeffs()
        height_loop = len(coef_v_loop)
        coef_blk[-1:-height_loop - 1:-1, col_count] = coef_v_loop[::-1]
        col_count -= 1

    return coef_blk


def coef2poly_2d(coef_blk, symbol_h, symbol_v):
    """
    build sympy polynomial from the coefficient data block (2D)
    :param coef_blk: the 2D coefficient data block. The upper left corner corresponds
            to the coefficient that has the highest power w.r.t. both symbol_h and symbol_v
    :param symbol_h: symbol used for the horizontal direction
    :param symbol_v: symbol used for the vertical direction
    :return:
    """
    max_degree_v, max_degree_h = np.array(coef_blk.shape) - 1
    poly = 0
    for h_count in range(max_degree_h + 1):
        for v_count in range(max_degree_v + 1):
            poly += coef_blk[v_count, h_count] * symbol_h ** (max_degree_h - h_count) * \
                    symbol_v ** (max_degree_v - v_count)
    return poly


def coef2poly_3d(coef_blk, symbol_h, symbol_v, symbol_d):
    """
    build sympy polynomial from the coefficient data block (3D)
    :param coef_blk: the 3D coefficient data block.
            The upper left corner corresponds to the coefficient that has
            the highest power w.r.t. symbol_h, symbol_v, and symbol_d
    :param symbol_h: symbol used for the horizontal direction
    :param symbol_v: symbol used for the vertical direction
    :param symbol_d: symbol used for the depth direction
    :return:
    """
    max_degree_v, max_degree_h, max_degree_d = np.array(coef_blk.shape) - 1
    poly = 0
    for d_count in range(max_degree_d + 1):
        for h_count in range(max_degree_h + 1):
            for v_count in range(max_degree_v + 1):
                poly += coef_blk[v_count, h_count, d_count] * \
                        symbol_h ** (max_degree_h - h_count) * \
                        symbol_v ** (max_degree_v - v_count) * \
                        symbol_d ** (max_degree_d - d_count)
    return poly


def compute_coef_yz(x_val, coef_3d):
    """
    compute the 2D polynoimal coefficients for a given x
    :param x_val: value of x
    :param coef_3d: the original 3D polynomials
    :return:
    """
    coef_yz = np.zeros((coef_3d.shape[0], coef_3d.shape[2]), dtype=coef_3d.dtype)
    max_degree_x = coef_3d.shape[1] - 1
    for x_power in range(max_degree_x + 1):
        coef_yz += coef_3d[:, x_power, :] * x_val ** (max_degree_x - x_power)

    return coef_yz


def compute_coef_xz(y_val, coef_3d):
    """
    compute the 2D polynoimal coefficients for a given x
    :param x_val: value of x
    :param coef_3d: the original 3D polynomials
    :return:
    """
    coef_xz = np.zeros((coef_3d.shape[1], coef_3d.shape[2]), dtype=coef_3d.dtype)
    max_degree_y = coef_3d.shape[0] - 1
    for y_power in range(max_degree_y + 1):
        coef_xz += coef_3d[y_power, :, :] * y_val ** (max_degree_y - y_power)

    return coef_xz


def compute_coef_xy(z_val, coef_3d):
    """
    compute the 2D polynoimal coefficients for a given x
    :param x_val: value of x
    :param coef_3d: the original 3D polynomials
    :return:
    """
    coef_xy = np.zeros((coef_3d.shape[0], coef_3d.shape[1]), dtype=coef_3d.dtype)
    max_degree_z = coef_3d.shape[2] - 1
    for z_power in range(max_degree_z + 1):
        coef_xy += coef_3d[:, :, z_power] * z_val ** (max_degree_z - z_power)

    return coef_xy


def eliminate_duplicate_roots_3d(all_roots1, all_roots2, all_roots3):
    total_roots = all_roots1.size
    flags = np.ones(total_roots, dtype=bool)
    for loop_outer in range(total_roots - 1):
        root1 = all_roots1[loop_outer]
        root2 = all_roots2[loop_outer]
        root3 = all_roots3[loop_outer]
        # compute the difference
        flags[loop_outer + 1 +
              np.where(np.sqrt((root1 - all_roots1[loop_outer + 1:]) ** 2 +
                               (root2 - all_roots2[loop_outer + 1:]) ** 2 +
                               (root3 - all_roots3[loop_outer + 1:]) ** 2) < 1e-2)[0]] = False

    return all_roots1[flags], all_roots2[flags], all_roots3[flags]


def check_error_3d(coef, x_val, y_val, z_val):
    val = 0
    max_degree_y, max_degree_x, max_degree_z = np.array(coef.shape) - 1
    for x_count in range(max_degree_x + 1):
        for y_count in range(max_degree_y + 1):
            for z_count in range(max_degree_z + 1):
                val += coef[y_count, x_count, z_count] * x_val ** (max_degree_x - x_count) * \
                       y_val ** (max_degree_y - y_count) * z_val ** (max_degree_z - z_count)
    return val


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
        num_samples = (max_poly_degree + 1) * 8  # <= 8 is the over-sampling factor used to determined the poly coef.

        # randomly generate y-values
        y_vals = np.exp(1j * 2 * np.pi / num_samples * np.arange(num_samples))[:, np.newaxis]
        y_powers = np.reshape(np.arange(max_poly_degree + 1)[::-1], (1, -1), order='F')
        Y = ne.evaluate('y_vals ** y_powers')

        # compute resultant, which is the determinant of mtx.
        # it is a polynomial in terms of variable y
        func_resultant = sympy.lambdify(y, sympy.Matrix(mtx))
        det_As = np.array([linalg.det(np.array(func_resultant(y_roots_loop), dtype=complex))
                           for y_roots_loop in y_vals.squeeze()], dtype=complex)
        coef_resultant = linalg.lstsq(Y, det_As)[0]

        y_roots_all = np.roots(coef_resultant)

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
            for roots_loop in x_roots_loop:
                x_roots.append(roots_loop)
            for roots_loop in np.tile(y_roots_loop, x_roots_loop.size):
                y_roots.append(roots_loop)

    x_roots, y_roots = np.array(x_roots).flatten('F'), np.array(y_roots).flatten('F')
    # x_roots, y_roots = eliminate_duplicate_roots_2d(x_roots, y_roots)
    # validate based on the polynomial values of the other polynomila
    # that is not used in the last step to get the roots
    poly_val = np.abs(
        check_error_2d(coef_validate / linalg.norm(coef_validate.flatten()),
                       x_roots, y_roots))

    # if the error is 2 orders larger than the smallest error, then we discard the root
    # print(poly_val)
    valid_idx = poly_val < tol
    x_roots = x_roots[valid_idx]
    y_roots = y_roots[valid_idx]

    return x_roots, y_roots


if __name__ == '__main__':
    '''
    test cases
    '''
    # first when one filter only depends on one variable but independent of the others
    # coef1 = np.random.randn(1, 3, 1) + 1j * np.random.randn(1, 3, 1)
    # coef2 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # coef3 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # x_roots, y_roots, z_roots = find_roots_3d(coef1, coef2, coef3)
    # print(np.abs(check_error_3d(coef1, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef2, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef3, x_roots, y_roots, z_roots)))
    #
    # coef1 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # coef2 = np.random.randn(3, 1, 1) + 1j * np.random.randn(3, 1, 1)
    # coef3 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # x_roots, y_roots, z_roots = find_roots_3d(coef1, coef2, coef3)
    # print(np.abs(check_error_3d(coef1, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef2, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef3, x_roots, y_roots, z_roots)))
    #
    # coef1 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # coef2 = np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)
    # coef3 = np.random.randn(1, 1, 3) + 1j * np.random.randn(1, 1, 3)
    # x_roots, y_roots, z_roots = find_roots_3d(coef1, coef2, coef3)
    # print(np.abs(check_error_3d(coef1, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef2, x_roots, y_roots, z_roots)))
    # print(np.abs(check_error_3d(coef3, x_roots, y_roots, z_roots)))

    # now the general cases
    coef1 = np.random.randn(2, 3, 1) + 1j * np.random.randn(2, 3, 1)
    coef2 = np.random.randn(3, 2, 1) + 1j * np.random.randn(3, 2, 1)
    coef3 = np.random.randn(1, 2, 3) + 1j * np.random.randn(1, 2, 3)
    x_roots, y_roots, z_roots = find_roots_3d(coef1, coef2, coef3)
    print(np.abs(check_error_3d(coef1, x_roots, y_roots, z_roots)))
    print(np.abs(check_error_3d(coef2, x_roots, y_roots, z_roots)))
    print(np.abs(check_error_3d(coef3, x_roots, y_roots, z_roots)))
