#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++

"""
Cython function for create contours around labels that is extremely fast and low memory.
Exposed function is contours_around_labels.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

cimport cython
from libc.math cimport fabs, isnan, NAN

DEF MARKED   = 0x01
DEF R_EDGE   = 0x10
DEF T_EDGE   = 0x20
DEF L_EDGE   = 0x40
DEF B_EDGE   = 0x80
DEF ANY_EDGE = 0xF0
DEF INSIDE  = 0
DEF EDGE    = 1
DEF OUTSIDE = 2

ctypedef Py_ssize_t ssize_t

# Allows looking up edge masks by index
cdef unsigned char* EDGES = [R_EDGE, T_EDGE, L_EDGE, B_EDGE] # EDGES[e] = 1 << (e + 4)

# When following a straight edge this is the direction that edge is going
cdef ssize_t* NEXT_Y = [1, 0, -1, 0]
cdef ssize_t* NEXT_X = [0, -1, 0, 1] # NEXT_X[e] = NEXT_Y[3-e]
# When rounding an inside corner this is the pixel we need to change to
cdef ssize_t* CORNER_Y = [1, 1, -1, -1]
cdef ssize_t* CORNER_X = [1, -1, -1, 1] # CORNER_X[e] = CORNER_Y[(e+3)%4]
# The actual point value (relative to the pixel coordinate) depending on the edge
cdef double* POINT_X = [0.95, 0.5 , 0.05, 0.5 ] # OTHER_X[e] = .5 + NEXT_Y[e]*.45
cdef double* POINT_Y = [0.5,  0.95, 0.5 , 0.05] # OTHER_Y[e] = .5 - NEXT_X[e]*.45 = OTHER_X[(e+1)%4]

cdef inline bint collinear(double[:,::1] a, ssize_t i, ssize_t j, ssize_t k, double thresh=1e-10) nogil:
    """Checks if 3 2D points are collinear."""
    cdef double x1 = a[i][0], y1 = a[i][1]
    cdef double x2 = a[j][0], y2 = a[j][1]
    cdef double x3 = a[k][0], y3 = a[k][1]
    return fabs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)) < thresh

cdef ssize_t contour_around(ssize_t i, ssize_t j, ssize_t e,
                            unsigned char[:,::1] data, double[:,::1] pts,
                            ssize_t x_off, ssize_t y_off) : #nogil:
    """
    Follows the contour around a set of edges starting at pixel i,j with edge e. The contour data is
    stored in the given buffer. The number of items used in the buffer is returned as a positive
    number (meaning rows [0,n) are valid). If the first item in the buffer is to be skipped, the
    value is returned negative (meaning rows [1,-n) are valid). If 0 is the contour needed more
    points than the buffer could store. Clears edges in the given data for any edge that is
    processed. The x and y values of each point are offset using the given offsets.
    """
    cdef ssize_t i_start = i, j_start = j, e_start = e, n = 0, nmax = pts.shape[0]
    cdef bint replace = False
    cdef int last = -1 # 0 = inside, 1 = edge, 2 = outside
    while n == 0 or i != i_start or j != j_start or e != e_start:
        # Look for next edge first around corner on same pixel
        if data[i,j] & EDGES[(e+1)%4]:
            # outside corner: same pixel, next edge
            e = (e+1)%4
            replace = last == OUTSIDE; last = INSIDE
        elif data[i+NEXT_Y[e], j+NEXT_X[e]] & EDGES[e]:
            # edge: same edge, next pixel
            i += NEXT_Y[e]; j += NEXT_X[e]
            replace = last == EDGE; last = EDGE
        else:
            # inside corner
            i += CORNER_Y[e]; j += CORNER_X[e]; e = (e+3)%4
            #assert(data[i,j] & EDGES[e])
            replace = last == INSIDE; last = OUTSIDE
        # add the point and clear the edge
        # the last point added is replaced if going it would create collinear points
        # however this does not handle collinearities at the beginning/end (that is handled later)
        if replace: n -= 1
        elif n >= nmax: return 0
        pts[n,0] = j + POINT_X[e] + x_off
        pts[n,1] = i + POINT_Y[e] + y_off
        n += 1
        data[i,j] &= ~EDGES[e]
    if n >= 3 and collinear(pts, n-2, n-1, 0): n -= 1
    if n >= 3 and collinear(pts, n-1, 0, 1): return -n
    return n

cdef void calc_edges(unsigned char[:,::1] d) nogil:
    """Takes a set of mask data and calculates the edges for it in-place."""
    cdef ssize_t h = d.shape[0], w = d.shape[1], h1 = h-1, w1 = w-1, i, j
    cdef bint hs = h != 1, ws = w != 1
    cdef unsigned char side
    # corners
    if d[0,0]:  d[0,0]  =MARKED|L_EDGE|B_EDGE|(0 if ws and d[0,   1] else R_EDGE)|(0 if hs and d[  1, 0] else T_EDGE)
    if d[0,w1]: d[0,w1] =MARKED|R_EDGE|B_EDGE|(0 if ws and d[0, w-2] else L_EDGE)|(0 if hs and d[  1,w1] else T_EDGE)
    if d[h1,0]: d[h1,0] =MARKED|L_EDGE|T_EDGE|(0 if ws and d[h1,  1] else R_EDGE)|(0 if hs and d[h-2, 0] else B_EDGE)
    if d[h1,w1]:d[h1,w1]=MARKED|R_EDGE|T_EDGE|(0 if ws and d[h1,w-2] else L_EDGE)|(0 if hs and d[h-2,w1] else B_EDGE)
    hs = not hs; ws = not ws
    # i == 0
    for j in xrange(1, w1):
        if d[0,j]:
            side = MARKED | B_EDGE
            if not (d[0,j+1] & MARKED): side |= R_EDGE
            if hs or not (d[1,j  ] & MARKED): side |= T_EDGE
            if not (d[0,j-1] & MARKED): side |= L_EDGE
            d[0,j] = side
    # j == 0
    for i in xrange(1, h1):
        if d[i,0]:
            side = MARKED | L_EDGE
            if ws or not (d[i,  1] & MARKED): side |= R_EDGE
            if not (d[i+1,0] & MARKED): side |= T_EDGE
            if not (d[i-1,0] & MARKED): side |= B_EDGE
            d[i,0] = side
        # core
        for j in xrange(1, w1):
            if d[i,j]:
                side = MARKED
                if not (d[i,  j+1] & MARKED): side |= R_EDGE
                if not (d[i+1,j  ] & MARKED): side |= T_EDGE
                if not (d[i  ,j-1] & MARKED): side |= L_EDGE
                if not (d[i-1,j  ] & MARKED): side |= B_EDGE
                d[i,j] = side
        # j == w1
        if d[i,w1]:
            side = MARKED | R_EDGE
            if not (d[i+1,w1 ] & MARKED): side |= T_EDGE
            if ws or not (d[i,  w-2] & MARKED): side |= L_EDGE
            if not (d[i-1,w1 ] & MARKED): side |= B_EDGE
            d[i,w1] = side
    # i == h1
    for j in xrange(1, w1):
        if d[h1,j]:
            side = MARKED | T_EDGE
            if not (d[h1, j+1] & MARKED): side |= R_EDGE
            if not (d[h1, j-1] & MARKED): side |= L_EDGE
            if hs or not (d[h-2,j  ] & MARKED): side |= B_EDGE
            d[h1,j] = side

cdef list contours_around(unsigned char[:,::1] data, double[:,::1] buf, ssize_t x_off, ssize_t y_off):
    """
    Takes the mask data and calculates the edges of it. Then it searches through the data to the
    find edges and calls contour_around with the found edge. Repeats this until there are no edges
    left. Return a list of contours. The buffer is used by contour_around. Every contour is offset
    by the given x and y offsets.
    """
    cdef ssize_t h = data.shape[0], w = data.shape[1], i, j, e, n

    # Calculate the edges in the data in-place
    calc_edges(data)

    # Calculate the contours around the edges
    cdef list conts = [] # requires GIL
    for i in xrange(h):
        for j in xrange(w):
            if data[i,j] & ANY_EDGE:
                for e in xrange(4):
                    if data[i,j] & EDGES[e]:
                        n = contour_around(i, j, e, data, buf, x_off, y_off)
                        if n == 0: raise BufferError()
                        break
                else: raise RuntimeError()
                cont = buf.base[1:-n] if n < 0 else buf.base[0:n] # requires GIL
                conts.append(cont.copy()) # requires GIL
    return conts # requires GIL

def contours_around_labels(im, double z=NAN, dict conts=None):
    """
    Calculate the contours around all of the labels in an image. The image must be 2D/grayscale and
    be at least 3x3. The return values is a dictionary with keys for the labels and values that are
    lists of Nx2 arrays where each array is a list of points representing the contour. If a second
    argument is provided this will be the "z" coordinate value written to every point. In this case
    the arrays have 3 columns instead of 2. Finally, a dictionary of contours as returned by this
    function, can be given and it will be merged, in-place, with the new data.
    """
    assert(im.ndim == 2 and im.shape[0] > 2 and im.shape[1] > 2)
    from numpy import empty, uint8
    from scipy.ndimage import find_objects
    
    # Get the bounding boxes of each label
    objs = find_objects(im)
    
    # Pre-allocate buffer for contours lists
    # Have at least 512 points or 2*perimeter of largest bounding box
    buf_rows = max(512,
                   4*max((o[0].stop-o[0].start+o[1].stop-o[1].start) for o in objs if o is not None))
    cdef double[:,::1] buf = empty((buf_rows, 2 if isnan(z) else 3))
    if not isnan(z): buf[:,2] = z
    
    # Go through each label and get the contours around it
    cdef ssize_t x_off, y_off
    if conts is None: conts = {}
    for lbl,obj in enumerate(objs, 1):
        lst = conts.setdefault(lbl, [])
        if obj is None: continue # no contours for a missing label

        # Get the binary mask of the labels
        data = im[obj[0], obj[1]] == lbl

        # Calculate the contours
        x_off = obj[1].start; y_off = obj[0].start
        lst.extend(contours_around(data.astype(uint8, copy=False), buf, x_off, y_off))
        del data

    return conts
