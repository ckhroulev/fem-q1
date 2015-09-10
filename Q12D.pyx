# -*- mode: cython -*-
#cython: boundscheck=False
#cython: embedsignature=True
#cython: wraparound=False
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t double_t
ctypedef np.int32_t   int_t

cdef class Q1:
    """2D Q1 quadrilateral finite element.

    Precomputes values of element basis functions and derivatives of element basis functions
    when created and uses table lookup during the assembly.

    Computes J^{-1} and det(J)*w once per element and uses table lookup later.

    Q1.chi[i,j] is the element basis function i at the quadrature point j.

    Q1.detJxW[j] is the determinant of the Jacobian at the quadrature point j
    times the corresponding quadrature weight.

    Q1.dphi_xy[i,j,k] is the derivative of the shape function i at the quadrature point j
    with respect to variable k. (In the (x,y) system.)
    """

    cdef public np.ndarray chi, detJxW, dphi_xy
    cdef public int n_pts, n_chi
    cdef double_t[:,:] J, J_inv, points
    cdef double_t[:] weights, xis, etas
    cdef np.ndarray dchi

    def __init__(self, quadrature):
        self.n_chi = 4
        self.n_pts = quadrature.n_pts
        self.weights = quadrature.weights
        self.points  = quadrature.pts
        self.xis   = np.array([-1,  1,  1, -1], dtype='f8')
        self.etas  = np.array([-1, -1,  1,  1], dtype='f8')

        self.detJxW  = np.zeros(self.n_pts)
        self.dphi_xy = np.zeros((self.n_chi, self.n_pts, 2))
        self.chi     = np.zeros((self.n_chi, self.n_pts))
        self.dchi    = np.zeros((self.n_chi, self.n_pts, 2))
        self.J       = np.zeros((2,2))
        self.J_inv   = np.zeros((2,2))

        # precompute values of element basis functions at all quadrature points
        for i in xrange(self.n_chi):
            for j in xrange(self.n_pts):
                self.chi[i,j] = self._chi(i, quadrature.pts[j])

        # precompute values of derivatives of element basis functions at all quadrature points
        for i in xrange(self.n_chi):
            for j in xrange(self.n_pts):
                self.dchi[i,j,:] = self._dchi(i, quadrature.pts[j])

    def _chi(self, int i, pt):
        """Element basis function i at a point pt."""
        xi,eta = pt
        return 0.25 * (1.0 + self.xis[i]*xi) * (1.0 + self.etas[i]*eta)

    def _dchi(self, int i, pt):
        """Derivatives of the element basis function i at a point pt."""
        xi,eta = pt
        dchi_dxi   = 0.25 *   self.xis[i] * (1.0 + self.etas[i] * eta)
        dchi_deta  = 0.25 *  self.etas[i] * (1.0 +  self.xis[i] *  xi)

        return [dchi_dxi, dchi_deta]

    def reset(self, double_t[:,:] pts):
        cdef double_t[:] detJxW = self.detJxW, weights = self.weights, x = pts[:,0], y = pts[:,1]
        cdef double_t[:,:,:] dchi = self.dchi, dphi_xy = self.dphi_xy
        cdef double_t[:,:] points = self.points, J = self.J, J_inv = self.J_inv
        cdef double_t dx, dy, xi, eta, detJ, tmp

        # precompute phi_x, phi_y, and det(J)*w at all quadrature points
        for i in xrange(self.n_pts):
            xi  = points[i,0]
            eta = points[i,1]

            J[0,0] = 0.25 * (eta*(-x[3]+x[2]-x[1]+x[0]) - x[3]+x[2]+x[1]-x[0])
            J[0,1] = 0.25 * (eta*(-y[3]+y[2]-y[1]+y[0]) - y[3]+y[2]+y[1]-y[0])
            J[1,0] = 0.25 * ( xi*(-x[3]+x[2]-x[1]+x[0]) + x[3]+x[2]-x[1]-x[0])
            J[1,1] = 0.25 * ( xi*(-y[3]+y[2]-y[1]+y[0]) + y[3]+y[2]-y[1]-y[0])

            detJ = J[0,0] * J[1,1] - J[0,1] * J[1,0]
            tmp = 1.0 / detJ
            J_inv[0,0] =  J[1,1] * tmp
            J_inv[0,1] = -J[0,1] * tmp
            J_inv[1,0] = -J[1,0] * tmp
            J_inv[1,1] =  J[0,0] * tmp

            for j in xrange(self.n_chi):
                dphi_xy[j,i,0] = dchi[j,i,0] * J_inv[0,0] + dchi[j,i,1] * J_inv[0,1]
                dphi_xy[j,i,1] = dchi[j,i,0] * J_inv[1,0] + dchi[j,i,1] * J_inv[1,1]

            detJxW[i] = detJ * weights[i]

cdef class Q1EquallySpaced(Q1):
    """Elements equallly spaced in both directions."""
    cdef int initialized

    def __init__(self, q):
        Q1.__init__(self,q)
        initialized = 0

    def reset(self, double_t[:,:] pts):
        cdef double_t[:] x = pts[:,0], y = pts[:,1], detJxW = self.detJxW, weights = self.weights
        cdef double_t[:,:,:] dchi = self.dchi, dphi_xy = self.dphi_xy
        cdef double_t dx, dy, one_over_dx, one_over_dy

        if self.initialized == 1:
            return

        dx = x[1] - x[0]
        dy = y[2] - y[0]

        one_over_dx = 1.0 / dx
        one_over_dy = 1.0 / dy

        for j in xrange(self.n_pts):
            detJxW[j] = weights[j] * 4.0 * one_over_dx * one_over_dy

        for i in xrange(self.n_chi):
            for j in xrange(self.n_pts):
                dphi_xy[i,j,0] = dchi[i,j,0] * 2.0 * one_over_dx
                dphi_xy[i,j,1] = dchi[i,j,1] * 2.0 * one_over_dy

        self.initialized = 1

class Gauss1:
    def __init__(self):
        self.pts = np.array([[0.0, 0.0]])
        self.n_pts = 1
        self.weights = np.array([4.0])

class Gauss2x2:
    def __init__(self):
        # coordinates of quadrature points (sans the 1/sqrt(3)):
        xis   = np.array([-1,  1,  1, -1], dtype='f8')
        etas  = np.array([-1, -1,  1,  1], dtype='f8')

        self.n_pts = 4
        self.pts = np.vstack((xis, etas)).T / np.sqrt(3)
        self.weights = np.ones(self.n_pts)
