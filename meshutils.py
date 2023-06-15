import numpy as np
from scipy.spatial import Delaunay


def writeply(X,color,tri,filename):
    """
    Save out a triangulated mesh to a ply file
    
    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        vertex coordinates shape (3,Nvert)
        
    color : 2D numpy.array (dtype=float)
        vertex colors shape (3,Nvert)
        should be float in range (0..1)
        
    tri : 2D numpy.array (dtype=float)
        triangular faces shape (Ntri,3)
        
    filename : string
        filename to save to    
    """
    f = open(filename,"w");
    f.write('ply\n');
    f.write('format ascii 1.0\n');
    f.write('element vertex %i\n' % X.shape[1]);
    f.write('property float x\n');
    f.write('property float y\n');
    f.write('property float z\n');
    f.write('property uchar red\n');
    f.write('property uchar green\n');
    f.write('property uchar blue\n');
    f.write('element face %d\n' % tri.shape[0]);
    f.write('property list uchar int vertex_indices\n');
    f.write('end_header\n');

    C = (255*color).astype('uint8')
    
    for i in range(X.shape[1]):
        f.write('%f %f %f %i %i %i\n' % (X[0,i],X[1,i],X[2,i],C[0,i],C[1,i],C[2,i]));
    
    for t in range(tri.shape[0]):
        f.write('3 %d %d %d\n' % (tri[t,1],tri[t,0],tri[t,2]))

    f.close();


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