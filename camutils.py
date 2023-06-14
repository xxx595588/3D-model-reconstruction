import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi*rx/180.0
    ry = np.pi*ry/180.0
    rz = np.pi*rz/180.0

    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,-np.sin(ry)],[0,1,0],[np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = (Rz @ Ry @ Rx)
    
    return R 

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR,np.ones((1,npts))))
    
    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t-camL.t)

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((qL[:,i],-R @ qR[:,i])).T
        z,_,_,_ = np.linalg.lstsq(A,t,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5*(pts3L+pts3R)

    return pts3


def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)
    
    return residual.flatten()

def calibratePose(pts3,pts2,cam_init,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
        
    """

    # define our error function
    efun = lambda params: residuals(pts3,pts2,cam_init,params)        
    popt,_ = scipy.optimize.leastsq(efun,params_init)
    cam_init.update_extrinsics(popt)

    return cam_init


def decode(imprefix,start,threshold,imprefix_obj,threshold_obj):
    """
    Decode 10bit gray code pattern with the given difference
    threshold.  We assume the images come in consective pairs
    with filenames of the form <prefix><start>.png - <prefix><start+20>.png
    (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)

    Parameters
    ----------
    imprefix : str
      Prefix of where to find the images (assumed to be .png)

    start : int
      Image offset.  

    threshold : float
       Threshold for imprefix images 

    imprefix_obj : str
      Prefix of where to find the object images (assumed to be .png)

    threshold_obj : float
      Threshold for imprefix_obj images 

    Returns
    -------
    code : 2D numpy.array (dtype=float)
      Decoded 10 bit gray pattern
      
    mask : 2D numpy.array (dtype=float)
      The mask generated by imprefix images 

    mask_obj : 2D numpy.array (dtype=float)
      The mask generated by imprefix_obj images 
    
    """
    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
        
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
        
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
        
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]

    obj0 = plt.imread(imprefix_obj + '%02d' % (0)+'.png')
    obj1 = plt.imread(imprefix_obj + '%02d' % (1)+'.png')
    mask_obj = np.ones(imgs[0].shape)
    diff = np.sum((obj0-obj1)**2, axis=-1)
    mask_obj = mask_obj*(diff>threshold_obj)

    return code,mask,mask_obj


def values_helper(imprefixL_obj, imprefixR_obj, pts2L, pts2R):
    """
    Extracted points from left and right object image corresponding to
    left and right camera points. Take the average of extracted points.

    Parameters
    ----------
    imprefixL_obj : str
      prefix of where to find the object images which is corresponding to pts2L

    imprefixR_obj : str 
      prefix of where to find the object images which is corresponding to pts2R

    pts2L,pts2R : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (2,N)

    Returns
    -------
    color_value : 2D numpy.array (dtype=float)
      The average values of extracted points from left & right camera
    
    """
    imageL = plt.imread(imprefixL_obj + '%02d' % (1) +'.png')
    imageR = plt.imread(imprefixR_obj + '%02d' % (1) +'.png')
    l, r = [], []

    for i in range(pts2L.shape[1]):
        l.append(imageL[pts2L[1][i]][pts2L[0][i]])
        r.append(imageR[pts2R[1][i]][pts2R[0][i]])

    color_value = (np.array(l).T+np.array(r).T)/2

    return color_value


def reconstruct(imprefixL,imprefixR,threshold,imprefixL_obj,imprefixR_obj,threshold_obj,camL,camR):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view which have been encoded with a 20bit gray code.

    Parameters
    ----------
    imprefix : str
      Prefix for where the images are stored

    threshold : float
      Decodability threshold

    camL,camR : Camera
      Camera parameters

    Returns
    -------
    pts2L,pts2R,pts3 : 2D numpy.array (dtype=float)

    """

    CLh,maskLh,maskLh_obj = decode(imprefixL,0,threshold,imprefixL_obj,threshold_obj)
    CLv,maskLv,_ = decode(imprefixL,20,threshold,imprefixL_obj,threshold_obj)
    CRh,maskRh,maskRh_obj = decode(imprefixR,0,threshold,imprefixR_obj,threshold_obj)
    CRv,maskRv,_ = decode(imprefixR,20,threshold,imprefixR_obj,threshold_obj)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv*maskLh_obj
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv*maskRh_obj

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)

    color_value = values_helper(imprefixL_obj,imprefixR_obj,pts2L,pts2R)
    pts3 = triangulate(pts2L,camL,pts2R,camR)
    
    return pts2L,pts2R,pts3,color_value

def bounding_box(pts2L,pts2R,pts3,color_value,boxlimit):
    """
    Trim out all points which are not within the given bounding box.
    
    Parameters
    ----------
    pts2L,pts2R : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (2,N)

    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    color_value : 2D numpy.array (dtype=float)
      The average values of extracted points from left & right camera

    boxlimit : 1D numpy.array (dtype=float)
      Limit along the x, y, and z axis of a box containing the object

    Returns
    -------
    pts2L,pts2R,pts3,color_value : 2D numpy.array (dtype=float)

    """
    def bbp_con(i):
        if pts3.T[i][0]<boxlimit[0] or pts3.T[i][0]>boxlimit[1]\
           or pts3.T[i][1]<boxlimit[2] or pts3.T[i][1]>boxlimit[3]\
           or pts3.T[i][2]<boxlimit[4] or pts3.T[i][2]>boxlimit[5]:
            return True
        return False
    
    bbp_drop = []

    for i in range(pts3.T.shape[0]):
        if bbp_con(i):
            bbp_drop.append(i)

    pts2L = np.delete(pts2L.T, bbp_drop, axis=0).T
    pts2R = np.delete(pts2R.T, bbp_drop, axis=0).T
    pts3 = np.delete(pts3.T, bbp_drop, axis=0).T
    color_value = np.delete(color_value.T, bbp_drop, axis=0).T

    return pts2L,pts2R,pts3,color_value


def mesh_smooth(tri,step,pts3):
    """
    Compute for each 3D point the average location of that points 
    neighbors in the mesh.

    Parameters
    ----------
    tri : Delaunay
      The mesh surface
    
    step : int
      Iteration of mesh smoothing

    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    Returns
    -------

    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    """
    for _ in range(step):
        neighbor_pts, neighbor_ind = tri.vertex_neighbor_vertices
        for i in range(pts3.shape[1]):
            pts3[:,i] = np.mean(pts3[:,neighbor_ind[neighbor_pts[i]:neighbor_pts[i+1]]],axis=1) 

    return pts3


def tri_pruning(tri,pts3,color_value,trithresh):
    """
    Remove triangles from the surface mesh that include edges 
    that are longer than the threshold.

    Parameters
    ----------
    tri : Delaunay
      The mesh surface
    
    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    color_value : 2D numpy.array (dtype=float)
      The average values of extracted points from left & right camera

    trithresh : float
      Longest allowed edge that can appear in the mesh

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    tri_sim : Delaunay
      The final mesh surface 

    color_value : 2D numpy.array (dtype=float)

    """
    def tp_con(p1, p2, p3):
        d = np.sum((pts3.T[tri.simplices[p1][p2]]-pts3.T[tri.simplices[p1][p3]])**2,axis=0)**(.5)
        return d

    tp_drop = []

    for i in range(tri.simplices.shape[0]):
        if tp_con(i,0,1)>trithresh or tp_con(i,0,2)>trithresh or tp_con(i,1,2)>trithresh:
            tp_drop.append(i)
        
    tri.simplices = np.delete(tri.simplices, tp_drop, axis=0)
    keep = np.unique(tri.simplices)
    map = np.zeros(pts3.shape[1])
    pts3 = pts3[:,keep]
    color_value = color_value[:,keep]
    map[keep] = np.arange(0,keep.shape[0])
    tri.simplices=map[tri.simplices]

    return pts3,tri.simplices,color_value


def mesh(pts2L,pts2R,pts3,boxlimit,step,color_value,trithresh):
    """
    Do the mesh process with Bounding Box Pruning, Triangle Pruning
    and mesh smoothing.

    Parameters
    ----------
    pts2L,pts2R : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (2,N)

    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    boxlimit : 1D numpy.array (dtype=float)
      Limit along the x, y, and z axis of a box containing the object

    step : int
      Iteration of mesh smoothing

    color_value : 2D numpy.array (dtype=float)
      The average values of extracted points from left & right camera

    trithresh : float
      Longest allowed edge that can appear in the mesh

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
      Coordinates of N points stored in a array of shape (3,N)

    tri_sim : Delaunay
      The final mesh surface
    
    color_value : 2D numpy.array (dtype=float)

    """
    pts2L,pts2R,pts3,color_value = bounding_box(pts2L,pts2R,pts3,color_value,boxlimit)
    tri = Delaunay(pts2L.T)
    pts3 = mesh_smooth(tri, step, pts3)
    pts3,tri_sim,color_value = tri_pruning(tri,pts3,color_value,trithresh)

    return pts3,tri_sim,color_value