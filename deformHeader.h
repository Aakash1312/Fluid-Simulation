#include <iostream>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeAdvect.h>
#include <openvdb/tools/PointAdvect.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/util/NullInterrupter.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/math/Transform.h>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <mutex>
#include <time.h>
#include "cmath"
// #include <Eigen/MatrixFunctions>

extern double factor;
Eigen::Matrix3d getR(Eigen::Matrix3d FE)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(FE, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d W = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    return W*V.transpose();
}
Eigen::Matrix3d getS(Eigen::Matrix3d FE)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(FE, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d W = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d D = svd.singularValues().asDiagonal();
    return V * D * V.transpose();
}

double spline2(double x)
{
    if (x < 0)
    {
        x *= -1.0*factor;
    }
    if (x < 0.5*factor)
    {
        return 1.0 * (4.0*x*x*x/(factor*factor*factor) - 4.0*x*x/(factor*factor) + 2.0/3.0);
    }
    if (x < 1.0*factor)
    {
        return 1.0* ((-8.0 * (x * x * x)/(6.0*factor*factor*factor)) + 4.0*x*x/(factor*factor) - 4.0*x/(factor) + 4.0/3.0);
    }
    return 0;
}
double getSplineGradient(double x)
{
    if (x >= 0)
    {
        if (x < 0.5*factor)
        {
            return 1.0 * (12.0*x*x/(factor*factor) - 8.0*x/factor);
        }
        else
        {
            if (x <= 1.0*factor)
            {
                return 1.0* ((-8.0 * (x * x)/(2.0*factor*factor)) + 8.0*x/factor - 4.0);
            }
        }

        return 0;
    }
    else
    {
        if (x > -0.5*factor)
        {
            return 1.0 * (-12.0*x*x/(factor*factor) - 8.0*x/factor);
        }
        else
        {
            if (x >= -1.0*factor)
            {
                return 1.0* ((8.0 * (x * x)/(2.0*factor*factor)) + 8.0*x/factor + 4.0);
            }
        }

        return 0;
    }
}

Eigen::Vector3d getGradW(openvdb::Coord c, openvdb::Vec3d p)
{
    // openvdb::Coord ipjk = openvdb::Coord(c.x()+1, c.y(), c.z());
    // openvdb::Coord ijpk = openvdb::Coord(c.x(), c.y()+1, c.z());
    // openvdb::Coord ijkp = openvdb::Coord(c.x()+1, c.y(), c.z()+1);
    // double x = -1 * getSplineGradient(p.x() - c.x())*spline2(c.y() - p.y())*spline2(c.z() - p.z());
    // double y = -1 * spline2(c.x() - p.x())*getSplineGradient(p.y() - c.y())*spline2(c.z() - p.z());
    // double z = -1 * spline2(c.x() - p.x())*spline2(c.y() - p.y())*getSplineGradient(p.z() - c.z());

    double x = -1 * getSplineGradient(p.x() - c.x() - 0.5)*spline2(0.5+c.y() - p.y())*spline2(0.5+c.z() - p.z());
    double y = -1 * spline2(0.5+c.x() - p.x())*getSplineGradient(p.y() - c.y() - 0.5)*spline2(0.5+c.z() - p.z());
    double z = -1 * spline2(0.5+c.x() - p.x())*spline2(0.5+c.y() - p.y())*getSplineGradient(p.z() - c.z() - 0.5);

    return Eigen::Vector3d(x,y,z);
    // return Eigen::Vector3d((spline2(ipjk.x() - p.x()) - spline2(c.x() - p.x())) * spline2(c.y() - p.y()) * spline2(c.z() - p.z())/dx, (spline2(ijpk.y() - p.y()) - spline2(c.y() - p.y())) * spline2(c.x() - p.x()) * spline2(c.z() - p.z())/dx, (spline2(ijkp.z() - p.z()) - spline2(c.z() - p.z())) * spline2(c.y() - p.y()) * spline2(c.x() - p.x())/dx);
}

Eigen::Matrix3d getDelFE(Eigen::Vector3d gradW, Eigen::Matrix3d FE, int i)
{
    // return FE.transpose() * getSplineGradient(xi, p);
    Eigen::Vector3d f = gradW.transpose() * FE;
    Eigen::Matrix3d m;
    if (i == 0)
    {
        m << f(0), f(1), f(2),
             0, 0, 0,
             0, 0, 0;
    }
    if (i == 1)
    {
        m << 0, 0, 0,
             f(0), f(1), f(2),
             0, 0, 0;
    }
    if (i == 2)
    {
        m << 0, 0, 0,
             0, 0, 0,
             f(0), f(1), f(2);
    }
    return m;

}
Eigen::Matrix3d getDelR(Eigen::Matrix3d S, Eigen::Matrix3d R, Eigen::Matrix3d dF)
{
    Eigen::Matrix3d rhs = R.transpose() * dF - dF.transpose() * R;
    Eigen::Vector3d v(rhs(0,1), rhs(0,2), rhs(1,2));
    Eigen::Matrix3d rdr;
    Eigen::Matrix3d m;
    m << S(0,0) + S(1,1), S(1,2), -1*S(0,2),
         S(1,2), S(0,0) + S(2,2), S(0,1),
         -1*S(0,2), S(0,1), S(1,1) + S(2,2);
    Eigen::Vector3d x = m.colPivHouseholderQr().solve(v);
    rdr << 0, x(0), x(1),
           -1*x(0), 0, x(2),
           -1*x(1), -1*x(2), 0;
    return R*rdr;
}
Eigen::Matrix<double, 9, 9> getdJF(Eigen::Matrix3d FE)
{
    Eigen::Matrix<double, 9, 9> m;
    // m << 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //      0, 0, -1*0, -1*0, 0, 0, 0, -1*0, 0,
    //      0, -1*0, 0, 0, 0, -1*0, -1*0, 0, 0,
    //      0, -1*0, 0, 0, 0, -1*0, -1*0, 0, 0,
    //      0, 0, 0, 0, 0, 0, 0, 0, 0,
    //      0, 0, -1*0, -1*0, 0, 0, 0, -1*0, 0,
    //      0, 0, -1*0, -1*0, 0, 0, 0, -1*0, 0,
    //      0, -1*0, 0, 0, 0, -1*0, -1*0, 0, 0,
    //      0, 0, 0, 0, 0, 0, 0, 0, 0;
    m << 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, FE(2,2), -1*FE(2,1), -1*FE(2,2), 0, FE(2,0), FE(2,1), -1*FE(2,0), 0,
         0, -1*FE(1,2), FE(1,1), FE(1,2), 0, -1*FE(1,0), -1*FE(1,1), FE(1,0), 0,
         0, -1*FE(2,2), FE(2,1), FE(2,2), 0, -1*FE(2,0), -1*FE(2,1), FE(2,0), 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, FE(0,2), -1*FE(0,1), -1*FE(0,2), 0, FE(0,0), FE(0,1), -1*FE(0,0), 0,
         0, FE(1,2), -1*FE(1,1), -1*FE(1,2), 0, FE(1,0), FE(1,1), -1*FE(1,0), 0,
         0, -1*FE(0,2), FE(0,1), FE(0,2), 0, -1*FE(0,0), -1*FE(0,1), FE(0,0), 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0;
    return m;
}

// Eigen::Matrix3d doubleDot42(Eigen::Matrix<double, 9, 9> m1, Eigen::Matrix3d m2)
// {
//     Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
//     for (int i = 0; i < 3; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//         {
//             for (int k = 0; k < 3; ++k)
//             {
//                 for (int l = 0; l < 3; ++l)
//                 {
//                     int indexi = i*3 + k;
//                     int indexj = j*3 + l;
//                     result(i,j) += m1(indexi, indexj) * m2(k,l);
//                 }
//             }
//         }
//     }
//     return result;
// }

Eigen::Matrix3d doubleDot42(Eigen::Matrix<double, 9, 9> m1, Eigen::Matrix3d m2)
{
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < 3; ++l)
                {
                    int indexi = i*3 + k;
                    int indexj = j*3 + l;
                    result(k,l) += m1(indexi, indexj) * m2(i,j);
                }
            }
        }
    }
    return result;
}

double doubleDot22(Eigen::Matrix3d m1, Eigen::Matrix3d m2)
{
    double result = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            result += m1(i,j) * m2(i,j);
        }
    }
    return result;
}

Eigen::Matrix3d getJFmt(Eigen::Matrix3d F)
{
    Eigen::Matrix3d result;
    result <<   F(1,1)*F(2,2) - F(1,2)*F(2,1), F(1,2)*F(2,0) - F(1,0)*F(2,2), F(1,0)*F(2,1) - F(1,1)*F(2,0),
                F(0,2)*F(2,1) - F(0,1)*F(2,2), F(0,0)*F(2,2) - F(0,2)*F(2,0), F(0,1)*F(2,0) - F(0,0)*F(2,1),
                F(0,1)*F(1,2) - F(0,2)*F(1,1), F(0,2)*F(1,0) - F(0,0)*F(1,2), F(0,0)*F(1,1) - F(0,1)*F(1,0);
    // if (F.determinant() != 1)
    // {
        // std::cout << F.determinant() << std::endl;
        // std::cout << result*F.transpose() << std::endl;
    // }
    return result;
}

Eigen::Matrix3d dPsydFdF(Eigen::Vector3d gradW, Eigen::Matrix3d F, Eigen::Matrix3d R, Eigen::Matrix3d S, double lambda, double mu, double J, int i)
{
    Eigen::Matrix3d dF = getDelFE(gradW, F, i);
    Eigen::Matrix3d dR = getDelR(S, R, dF);
    Eigen::Matrix3d JFmt = getJFmt(F);
    Eigen::Matrix3d dJFmt = doubleDot42(getdJF(F), dF);
    // std::cout << JFmt << std::endl;
    return 2*mu*dF - 2*mu*dR + lambda*JFmt*doubleDot22(JFmt, dF) + lambda*(J - 1)*dJFmt;
}

Eigen::Matrix3d getdPsydx2(openvdb::Coord xi, openvdb::Coord xj, openvdb::Vec3d p, Eigen::Matrix3d F, Eigen::Matrix3d FP, double lambda0, double mu0, double epsilon)
{
    double Jp = FP.determinant();
    double mu = mu0 * std::exp(epsilon*(1 - Jp));
    double lambda = lambda0 * std::exp(epsilon*(1 - Jp));

    Eigen::Vector3d gradW = getGradW(xi,p);
    Eigen::Vector3d gradWj = getGradW(xj,p);
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d R = getR(F);
    Eigen::Matrix3d S = getS(F);
    double J = F.determinant();
    Eigen::Matrix3d Ft = F.transpose();
    Eigen::Vector3d v1 = dPsydFdF(gradWj, F, R, S, lambda, mu, J, 0)*Ft*gradW;
    Eigen::Vector3d v2 = dPsydFdF(gradWj, F, R, S, lambda, mu, J, 1)*Ft*gradW;
    Eigen::Vector3d v3 = dPsydFdF(gradWj, F, R, S, lambda, mu, J, 2)*Ft*gradW;
    Eigen::Matrix3d result;
    result <<   v1(0), v2(0), v3(0),
                v1(1), v2(1), v3(1),
                v1(2), v2(2), v3(2);
    return result;
}
Eigen::Matrix3d getSigma(double mu0, double lambda0, double epsilon, Eigen::Matrix3d FE, Eigen::Matrix3d FP)
{
    double Jp = FP.determinant();
    double mu = mu0 * std::exp(epsilon*(1 - Jp));
    double lambda = lambda0 * std::exp(epsilon*(1 - Jp));
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(FE, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d R = getR(FE);
    double Je = FE.determinant();
    // if (Jp < 1)
    // {
    //     std::cout << "Jp " << Jp << std::endl;
    // }
    // if (lambda > 1 || mu > 1)
    // {
    //     std::cout << lambda << " " << mu << " " << std::exp(epsilon*(1 - Jp)) << " " << Jp << std::endl;
    // }
    // if (mu > 1)
    // {
    //     std::cout << "mu" << std::endl;
    // }
    // if (lambda > 1)
    // {
    //     std::cout << "lambda" << std::endl;
    // }
    // if (Je > 1)
    // {
    //     std::cout << "Je" << std::endl;
    // }
    // std::cout << FE << std::endl;
    // std::cout << R * R.transpose() << std::endl;
    // std::cout << mu << " " << lambda << std::endl;
    // std::cout << 2*mu*(FE-R)*(FE.transpose()) + lambda*(Je - 1)*Je*(Eigen::Matrix3d::Identity()) << std::endl;
    // if (Jp >= 1)
    // {
        return 2*mu*(FE-R)*(FE.transpose()) + lambda*(Je - 1)*Je*(Eigen::Matrix3d::Identity());
    // }
    // else
    // {
    //     return Eigen::Matrix3d::Zero();
    // }
}
// Eigen::Matrix3d getGradV(openvdb::Coord c, openvdb::Vec3d p, double dx, openvdb::Vec3dGrid::Ptr velocity)
// {
//     openvdb::Coord ipjk = openvdb::Coord(c.x()+1, c.y(), c.z());
//     openvdb::Coord ijpk = openvdb::Coord(c.x(), c.y()+1, c.z());
//     openvdb::Coord ijkp = openvdb::Coord(c.x()+1, c.y(), c.z()+1);
//     openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
//     openvdb::Vec3d cv = vaccessor.getValue(c);
//     double cw = spline2(c.x() - p.x())*spline2(c.y() - p.y())*spline2(c.z() - p.z());
//     double x = (vaccessor.getValue(ipjk).x() - cv.x())*cw/dx;
//     double y = (vaccessor.getValue(ijpk).y() - cv.y())*cw/dx;
//     double z = (vaccessor.getValue(ijkp).z() - cv.z())*cw/dx;

//     return Eigen::Vector3d(x,y,z).asDiagonal();
//     // return Eigen::Vector3d((spline2(ipjk.x() - p.x()) - spline2(c.x() - p.x())) * spline2(c.y() - p.y()) * spline2(c.z() - p.z())/dx, (spline2(ijpk.y() - p.y()) - spline2(c.y() - p.y())) * spline2(c.x() - p.x()) * spline2(c.z() - p.z())/dx, (spline2(ijkp.z() - p.z()) - spline2(c.z() - p.z())) * spline2(c.y() - p.y()) * spline2(c.x() - p.x())/dx);
// }
// Eigen::MatrixXd HessianPsy()
// int main(int argc, char* argv[])
// {
//     openvdb::initialize();

//     Eigen::Matrix3d FE;
//     Eigen::Matrix3d FP;
//     FE = Eigen::Matrix3d::Identity();
//     FP = Eigen::Matrix3d::Identity();
// }
