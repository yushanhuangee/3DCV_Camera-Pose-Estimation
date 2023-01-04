import numpy as np
import cv2
from utils import *
import math

class EPnP:
    def __init__(self, cameraMatrix, points2D, points3D):
        #init_camera_parameters
        self.uc = cameraMatrix[0, 2]
        self.vc = cameraMatrix[1, 2]
        self.fu = cameraMatrix[0, 0]
        self.fv = cameraMatrix[1, 1]
        self.K=cameraMatrix

        self.points2D = points2D
        self.points3D = points3D
        self.n = points3D.shape[0]
    
    def choose_control_point(self):
        control_3d_points = []

        #pi = ai_0*cws0 + ai_1*cws1 + ai_2*cws2 + ai_3*cws3
        #first control point: centroid
        centroid = np.mean(self.points3D, axis=0).reshape((1,3))
        control_3d_points.append(centroid)
        #A = pwi-cw1
        Pwo = self.points3D - centroid
        PW0tPW0 = np.matmul(Pwo.T, Pwo)

        #find other 3 control points by calculatng eigenvalue
        '''
        control_3d_points.append([[1,0,0]])
        control_3d_points.append([[0,1,0]])
        control_3d_points.append([[0,0,1]])
        control_3d_points = np.asarray(control_3d_points)

        
        '''
        w, v = np.linalg.eig(PW0tPW0)
        #vTv=np.matmul(v.T, v)/self.n
        #sign_value = [1, -1, -1]
        

        for i in range(3):
            #control_3d_points.append(centroid + sign_value[i] *w[i]* vTv[:, i].T)
            control_3d_points.append(centroid + math.sqrt(w[i])*v[:, i].T)
        control_3d_points = np.asarray(control_3d_points)
        

        return control_3d_points
    
    def computeBaryCentricCoordinates(self, control_3d_points):

        control_3d_points = get_homo_from_x(control_3d_points).transpose((1, 0))
        Pwo = get_homo_from_x(self.points3D).transpose((1, 0))
        alpha = np.matmul(np.linalg.inv(control_3d_points), Pwo)

        return alpha
    
    def calculateM(self, alpha):
        alpha = alpha.transpose((1,0)) #(N,4)
        M = np.zeros((2*self.n, 12))        

        for i in range(self.n):
            uci = self.uc - self.points2D[i, 0]
            vci = self.vc - self.points2D[i, 1]
            for j in range(4):
                a = alpha[i,j]
                M[2*i, 3*j] = a*self.fu
                M[2*i, 3*j + 2] = a*uci
                M[2*i + 1, 3*j + 1] = a*self.fv
                M[2*i + 1, 3*j + 2] = a*vci
        return M

    def compute_rho(self, control_wcs, diff_pattern):
        
        control_point_diff = np.zeros((6,1))
        for i, (a, b)in enumerate(diff_pattern):
            c_diff = control_wcs[a] - control_wcs[b]
            control_point_diff[i] = np.matmul(c_diff.T, c_diff)
        return control_point_diff
    
    def compute_L6X10(self, vh, diff_pattern):
        DiffMat = np.zeros((18, 4))
        L6X10 = np.zeros((6,10))
        for i in range(6):
            a, b = diff_pattern[i]
            DiffMat[3*i:3*i+3, :] = vh[3*a:3*a+3,:]-vh[3*b:3*b+3,:]	
        for i in range(6):
            v1 = DiffMat[3*i:3*i+3, 0]
            v2 = DiffMat[3*i:3*i+3, 1]
            v3 = DiffMat[3*i:3*i+3, 2]
            v4 = DiffMat[3*i:3*i+3, 3]
            L6X10[i, :] = np.asarray([np.matmul(v1.T, v1), 2*np.matmul(v1.T, v2), np.matmul(v2.T, v2), 2*np.matmul(v1.T, v3), 2*np.matmul(v2.T, v3),
                                    np.matmul(v3.T, v3), 2*np.matmul(v1.T, v4), 2*np.matmul(v2.T, v4), 2*np.matmul(v3.T, v4), np.matmul(v4.T, v4)])

        return L6X10

    def findBetasN4(self, L6X10, rho):
        '''
        when X = B1V1+B2V2+B3V3+B4V4, solve LB= rho
        B11 = B1*B1;
        B12 = B1*B2;
        B13 = B1*B3;
        B14 = B1*B4;
        that is:
        B1 = sqrt(B11)
        B2 = B12/B11
        B3 = B13/B11
        B4 = B14/B11
        '''
        beta_4 = np.zeros((4, 1))
        L6X4 = np.stack((L6X10[:,0],L6X10[:,1], L6X10[:,3], L6X10[:, 6]), axis=1)
        Beta = np.matmul(np.linalg.pinv(L6X4), rho)

        if Beta[0] < 0:
            beta_4[0] = math.sqrt(-Beta[0])
            beta_4[1] = -Beta[1]/beta_4[0]
            beta_4[2] = -Beta[2]/beta_4[0]
            beta_4[3] = -Beta[3]/beta_4[0]
        else:
            beta_4[0] = math.sqrt(Beta[0])
            beta_4[1] = Beta[1]/beta_4[0]
            beta_4[2] = Beta[2]/beta_4[0]
            beta_4[3] = Beta[3]/beta_4[0]

        return beta_4

    def findBetasN3(self, L6X10, rho):
        '''
        when X = B1V1+B2V2+B3V3, solve LB= rho
        B11 = B1*B1;
        B12 = B1*B2;
        B22 = B2*B2;
        B13 = B1*B3;
        B23 = B2*B3;
        '''
        beta_3 = np.zeros((4, 1))
        L6X6 = L6X10[:, :6]
        Beta = np.matmul(np.linalg.inv(L6X6), rho)

        if Beta[0] < 0:
            beta_3[0] = math.sqrt(-Beta[0])
            beta_3[1] = math.sqrt(-Beta[2]) if Beta[2]<0 else 0
        else:
            beta_3[0] = math.sqrt(Beta[0])
            beta_3[1] = math.sqrt(Beta[2]) if Beta[2]>0 else 0
        if Beta[1]<0:
            beta_3[0] *= -1 
        beta_3[2] = Beta[3]/beta_3[0]

        return beta_3
    
    def findBetasN2(self, L6X10, rho):
        '''
        betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        betas_approx_2 = [B11 B12 B22                            ] 
        when X = B1V1+B2V2, solve LB= rho, solutions are B11,B12,B22
        B11 = B1*B1;
        B12 = B1*B2;
        B22 = B2*B2;
        B1 = sqrt(B11)
        B2 = sqrt(B22)
        '''
        beta_2 = np.zeros((4, 1))
        L6X3 = L6X10[:, :3]
        Beta = np.matmul(np.linalg.inv(np.matmul(L6X3.T, L6X3)), np.matmul(L6X3.T,rho))
        if Beta[0] < 0:
            beta_2[0] = math.sqrt(-Beta[0])
            beta_2[1] = math.sqrt(-Beta[2]) if Beta[2]<0 else 0
        else:
            beta_2[0] = math.sqrt(Beta[0])
            beta_2[1] = math.sqrt(Beta[2]) if Beta[2]>0 else 0
        if Beta[1]<0:
            beta_2[0] *= -1

        return beta_2
    
    def findBetasN1(self, L6X10, rho):
        v_2 = L6X10[:, 0]
        beta_1 = np.zeros((4,1))
        beta_1[0] = np.sum(np.sqrt(v_2)*np.sqrt(rho))/np.sum(v_2)

        return beta_1

    def computeGaussNewtonJacobian(self, L6X10, beta):
        #jacobian(6, 4)
        #beta(1,4)
        beta = beta.squeeze()
        
        L2J = np.asarray([[2*beta[0], 0, 0, 0], 
                        [beta[1], beta[0], 0, 0],
                        [0, 2*beta[1], 0, 0],
                        [beta[2], 0, beta[0], 0],
                        [0, beta[2], beta[1], 0],
                        [0, 0, 2*beta[2], 0],
                        [beta[3], 0, 0, beta[0]],
                        [0, beta[3], 0, beta[1]],
                        [0, 0, beta[3], beta[2]],
                        [0, 0, 0, 2*beta[3]]])
        jacobian = np.matmul(L6X10, L2J)

        return jacobian

    def computeResiduals(self, eigvector, betas, diff_pattern,control_3d_points):
        #eigvector(12,4)
        #CC(12, 1);
        betas = betas.squeeze()
        CC = betas[0] * eigvector[:, 0] + betas[1] * eigvector[:, 1] + betas[2] * eigvector[:, 2] + betas[3] *  eigvector[:, 3]
        residuals = np.zeros((6, 1))

        for i in range(6):
            a, b =diff_pattern[i]
            Ca = CC[3*a:3*a+3]
            Cb = CC[3*b:3*b+3]
            Wa = control_3d_points[a,:]
            Wb = control_3d_points[b,:]

            d1 = np.sum((Ca-Cb)**2)
            d2 = np.sum((Wa-Wb)**2)
            residuals[i] = d1-d2
        return residuals
         

    def doGaussNewtonOptimization(self, eigvector, L6x10, beta, diff_pattern,control_3d_points):
        #jacobian(6, 4)
        #residuals(6, 1)
        iterations_number = 5
        #JtJ(4, 4), JtJ_inv(4, 4);
        Vb = beta.transpose((1,0)) #(4,1)
        for i in range(iterations_number):
            jacobian = self.computeGaussNewtonJacobian(L6x10, Vb.transpose((1,0)))
            residuals = self.computeResiduals(eigvector, beta, diff_pattern, control_3d_points)
            JtJ = np.matmul(jacobian.T, jacobian) #(4,4)
            JtJ_inv = np.linalg.inv(JtJ)
            jacobian_res = np.matmul(jacobian.T, residuals) #(4,1)
            Vb = Vb - np.matmul(JtJ_inv, jacobian_res)   

        return Vb.transpose((1,0))

    def computeControlPointsUnderCameraCoord(self, control_ccs_v, betas):
        control_3d_points_ccs = np.zeros((4, 3))
        betas = betas[0]
        v = betas[0]*control_ccs_v[:,0] + betas[1] * control_ccs_v[:,1] + betas[2] * control_ccs_v[:,2] + betas[3] * control_ccs_v[:,3] #(12,)
        for i in range(4): 
            control_3d_points_ccs[i, :] = v[3*i:3*i+3]
        return control_3d_points_ccs

    def estimateRt(self, reference_3d_points_ccs):
        p0w = np.mean(self.points3D, axis=0) #(1,3)
        piw = self.points3D - p0w

        p0c = np.mean(reference_3d_points_ccs, axis=0)
        pic = reference_3d_points_ccs - p0c
        M = np.matmul(pic.T, piw)
        u, s, vh = np.linalg.svd(M)
        R = np.matmul(u, vh)
        detR = np.linalg.det(R)
        if detR<0:
            R=-R
        t = p0c.T - np.matmul(R, p0w.T)

        return R, t.reshape((3,1))

    def reprojectionError(self, R, t):
        #t (3,1)
        #R (3,3)
        p_3D = np.concatenate([self.points3D, np.ones((self.n,1))], axis=1).transpose((1,0)) #(4,N)
        H = np.concatenate([R, t], axis=1) #(3, 4)
        p_2D = self.K @ np.matmul(H,p_3D) #(3,N)

        error = 0.0
        H = np.concatenate([R, t], axis=1)
        for i in range(self.n):
            u, v = self.points2D[i]
            ue, ve = p_2D[:2,i]/p_2D[2,i]
            error += np.sqrt((np.sum((ue-u)**2)+np.sum((ve-v)**2)))
        return error/self.n


        '''
        error = 0.0
        X_t = self.points3D.T + t
        X_Rt = np.matmul(R, X_t)

        for i in range(self.n):
            X_c = X_Rt[0, i]
            Y_c = X_Rt[1, i]
            inv_Z_c = 1.0/X_Rt[2, i]
            ue = self.uc + self.fu*X_c* inv_Z_c
            ve = self.vc + self.fv*Y_c*inv_Z_c
            u, v = self.points2D[i]
            error += np.sqrt((np.sum((ue-u)**2)+np.sum((ve-v)**2)))
        
        return error/self.n
        '''

    
    def computeRt(self, control_ccs_v, new_beta, alpha, control_3d_points):
        control_3d_points_ccs = self.computeControlPointsUnderCameraCoord(control_ccs_v, new_beta) #(4,3)
        reference_3d_points_ccs = np.matmul(alpha.T, control_3d_points_ccs) #(N,3)
        #solve for sign
        if (reference_3d_points_ccs[0, 2] < 0):
            control_3d_points_ccs *= -1
            reference_3d_points_ccs *= -1
        R, t = self.estimateRt(reference_3d_points_ccs)
        return R, t, self.reprojectionError(R, t)




    def compute_Pose(self):
        control_3d_points = np.squeeze(self.choose_control_point())#(4, 3)
        alpha = self.computeBaryCentricCoordinates(control_3d_points) #alpha (4, N)
        M = self.calculateM(alpha)#Mx=0 M(2n, 12)
        #u, s, vh = np.linalg.svd(np.matmul(M.T,M))
        W, V = np.linalg.eig(np.matmul(M.T,M))
        idx = W.argsort()
        control_ccs_v = V[:, idx[:4]]

        #control_ccs_v = vh[:, -4:] #(12, 4)
        #control_ccs_v = control_ccs_v.reshape((4, 4, 3))

        diff_pattern = np.asarray([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


        #L * betas = rho
        L6X10 = self.compute_L6X10(control_ccs_v, diff_pattern) #(6,10) for N=4
        rho = self.compute_rho(control_3d_points, diff_pattern)# (6,1)#record the ditances between control points

        #calculate beta(4,4), repro_error(1,4)
        #RR (3,3) tt(3, 1) 
        #Rs(4,RR) ts(4,tt)=> used to record  
        Rs = np.zeros((4,3,3))  
        ts = np.zeros((4,3,1))
        error = np.zeros((4,1))
        #for N=[1,2,3,4]
        # N=1: B1 (1,4)
        beta_1 = self.findBetasN1(L6X10, rho).transpose((1,0))
        new_beta_1 = self.doGaussNewtonOptimization(control_ccs_v, L6X10, beta_1, diff_pattern, control_3d_points) #(1,4)
        Rs[0,:,:],ts[0,:,:] ,error[0] = self.computeRt(control_ccs_v, new_beta_1, alpha, control_3d_points)

        print("N=1:", error[0])

        # N=2: B1 B2 (B3, B4 = 0) (1,4)
        beta_2 = self.findBetasN2(L6X10, rho).transpose((1,0))
        new_beta_2 = self.doGaussNewtonOptimization(control_ccs_v, L6X10, beta_2, diff_pattern, control_3d_points)
        Rs[1,:,:],ts[1,:,:] ,error[1] = self.computeRt(control_ccs_v, new_beta_2, alpha, control_3d_points)

        print("N=2:", error[1])
        

        # N=3: B1 B2 B3 (B4 = 0) (1,4)
        beta_3 = self.findBetasN3(L6X10, rho).transpose((1,0))
        new_beta_3 = self.doGaussNewtonOptimization(control_ccs_v, L6X10, beta_3, diff_pattern, control_3d_points)
        Rs[2,:,:],ts[2,:,:] ,error[2] = self.computeRt(control_ccs_v, new_beta_3, alpha, control_3d_points)

        print("N=3:", error[2])

        # N=4: B1 B2 B3 B4 (1,4)
        beta_4 = self.findBetasN4(L6X10, rho).transpose((1,0))
        new_beta_4 = self.doGaussNewtonOptimization(control_ccs_v, L6X10, beta_4, diff_pattern, control_3d_points)
        Rs[3,:,:],ts[3,:,:] ,error[3] = self.computeRt(control_ccs_v, new_beta_4, alpha, control_3d_points)

        print("N=4:", error[3])

        #chosse R t with smallest repro_error

        print(np.argmin(error))
        idx = np.argmin(error)
        return Rs[idx, :,:], ts[idx, :,:]



        