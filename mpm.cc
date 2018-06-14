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
#include "deformHeader.h"

clock_t end;
double cpu_time_used;
clock_t start = clock();
typedef Eigen::Triplet<double> Triplet;
// Eigen::Matrix3d getSigma(double mu, double lambda, Eigen::Matrix3d FE, Eigen::Matrix3d FP)
// {
//     Eigen::JacobiSVD<Eigen::Matrix3d> svd(FE, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     Eigen::Matrix3d R = svd.matrixU();
//     double Je = FE.determinant();
//     return 2*mu*(FE-R)*(FE.transpose()) + lambda*(Je - 1)*Je*(Eigen::Matrix3d::Identity());
// }
double factor = 1.0;
// double spline(double x)
// {
//     if (x < 0)
//     {
//         x *= -1.0*factor;
//     }
//     if (x < 0.5*factor)
//     {
//         return 1.0 * (4.0*x*x*x/(factor*factor*factor) - 4.0*x*x/(factor*factor) + 2.0/3.0);
//     }
//     if (x <= 1.0*factor)
//     {
//         return 1.0* ((-8.0 * (x * x * x)/(6.0*factor*factor*factor)) + 4.0*x*x/(factor*factor) - 4.0*x/(factor) + 4.0/3.0);
//     }
//     return 0;
// }
double spline(double x)
{
    x -= 0.5;
    if (x < 0)
    {
        x *= -1.0*factor;
    }
    if (x < 0.5*factor)
    {
        return 1.0 * (4.0*x*x*x/(factor*factor*factor) - 4.0*x*x/(factor*factor) + 2.0/3.0);
    }
    if (x <= 1.0*factor)
    {
        return 1.0* ((-8.0 * (x * x * x)/(6.0*factor*factor*factor)) + 4.0*x*x/(factor*factor) - 4.0*x/(factor) + 4.0/3.0);
    }
    return 0;
}
bool isWithinBounds(openvdb::Coord xyz, int bound)
{
    if (fabs(xyz.x()) > bound || fabs(xyz.y()) > bound || fabs(xyz.z()) > bound)
    {
        return false;
    }
    return true;
}
bool isSolid(openvdb::Coord xyz, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::FloatGrid::Accessor saccessor = solidGrid->getAccessor();
    if (saccessor.getValue(xyz) == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// double spline(double x)
// {
//     if (x < 0)
//     {
//         x *= -1.0;
//     }
//     if (x < 1.0)
//     {
//         return 1.5 * (0.5*x*x*x - x*x + 2.0/3.0);
//     }
//     if (x < 2.0)
//     {
//         return 1.5* ((-1.0 * (x * x * x)/6.0) + x*x - 2.0*x + 4.0/3.0);
//     }
//     return 0;
// }

// double spline(double x)
// {
//     if (x < 0)
//     {
//         x *= -1.0;
//     }
//     if (x < 0.25)
//     {
//         return 1.5 * (0.5*64.0*x*x*x - 16.0*x*x + 2.0/3.0);
//     }
//     if (x < 0.5)
//     {
//         return 1.5* ((-64.0 * (x * x * x)/6.0) + 16*x*x - 13.0*x + 4.0/3.0);
//     }
//     return 0;
// }

openvdb::Vec3d getVelocity(openvdb::Coord c, openvdb::Vec3dGrid::Ptr velocity)
{
    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Vec3d vel = vaccessor.getValue(c);
    double u = (vel.x() + vaccessor.getValue(ipjk).x())/2.0;
    double v = (vel.y() + vaccessor.getValue(ijpk).y())/2.0;
    double w = (vel.z() + vaccessor.getValue(ijkp).z())/2.0;
    return openvdb::Vec3d(u, v, w);
}

void getStaggered(openvdb::Vec3dGrid::Ptr original, openvdb::math::Transform::Ptr trans)
{
    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create();
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0));
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    openvdb::Vec3dGrid::Accessor ovaccessor = original->getAccessor();
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
        {
            for (int z = -15; z <= 15; z++)
            {
                openvdb::Coord c = openvdb::Coord(x, y, z);
                openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
                openvdb::Vec3d vel = vaccessor.getValue(c);
                openvdb::Vec3d ovel = ovaccessor.getValue(c);
                openvdb::Vec3d ipv = vaccessor.getValue(ipjk);
                openvdb::Vec3d jpv = vaccessor.getValue(ijpk);
                openvdb::Vec3d kpv = vaccessor.getValue(ijkp);
                vaccessor.setValue(ipjk, openvdb::Vec3d(2*ovel.x() - vel.x(), ipv.y(), ipv.z()));
                vaccessor.setValue(ijpk, openvdb::Vec3d(jpv.x(), 2*ovel.y() - vel.y(), jpv.z()));
                vaccessor.setValue(ijkp, openvdb::Vec3d(kpv.x(), kpv.y(), 2*ovel.z() - vel.z()));
            }
        }
    }
    original = vels -> deepCopy();
    return;
}

void getUnstaggered(openvdb::Vec3dGrid::Ptr original, openvdb::math::Transform::Ptr trans)
{
    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create();
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0));
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
        {
            for (int z = -15; z <= 15; z++)
            {
                openvdb::Coord c = openvdb::Coord(x, y, z);
                vaccessor.setValue(c, getVelocity(c, original));
            }
        }
    }
    original = vels -> deepCopy();
    return;
}
openvdb::Vec3d clampedCatmullRom(openvdb::Vec3d c, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();

    double u = 0;
    double v = 0;
    double w = 0;

    double weight = 0;

    openvdb::Vec3d velc = vaccessor.getValue(openvdb::Coord(fcx, fcy, fcz));

    double minu = velc.x();
    double minv = velc.y();
    double minw = velc.z();
    double maxu = velc.x();
    double maxv = velc.y();
    double maxw = velc.z();

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                if (isWithinBounds(openvdb::Coord(x, y, z), 13))
                {
                    // velc = vaccessor.getValue(openvdb::Coord(x, y, z));
                    velc = getVelocity(openvdb::Coord(x, y, z), velocity);

                    minu = velc.x() < minu ? velc.x() : minu;
                    minv = velc.y() < minv ? velc.y() : minv;
                    minw = velc.z() < minw ? velc.z() : minw;

                    maxu = velc.x() > maxu ? velc.x() : maxu;
                    maxv = velc.y() > maxv ? velc.y() : maxv;
                    maxw = velc.z() > maxw ? velc.z() : maxw;

                    double mass = faccessor.getValue(openvdb::Coord(x, y, z));
                    // if (mass > 0)
                    // {
                        double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);//should it be mass*?
                        weight += cw;
                        u += velc.x() * cw;
                        v += velc.y() * cw;
                        w += velc.z() * cw;
                // }
                }
            }
        }
    }
    // weight = 1.0;//experiment
    if (weight != 0)
    {
        u /= weight;
        v /= weight;
        w /= weight;
    }
    else
    {
        return openvdb::Vec3d(0, 0, 0);
    }

    // u = u > maxu ? maxu : u;
    // v = v > maxv ? maxv : v;
    // w = w > maxw ? maxw : w;

    // u = u < minu ? minu : u;
    // v = v < minv ? minv : v;
    // w = w < minw ? minw : w;

    return openvdb::Vec3d(u, v, w);
}


openvdb::Vec3d CatmullRomFLIP(openvdb::Vec3d c, openvdb::Vec3d vel, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::Vec3dGrid::Ptr velBeforeUpdate, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Vec3dGrid::Accessor vpaccessor = velBeforeUpdate->getAccessor();
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();
    double weight = 0;
    openvdb::Vec3d velc;
    openvdb::Vec3d velp;
    openvdb::Vec3d delta(0, 0, 0);

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                if(isWithinBounds(openvdb::Coord(x, y, z), 13))
                {
                    // velc = vaccessor.getValue(openvdb::Coord(x, y, z));
                    // velp = vpaccessor.getValue(openvdb::Coord(x, y, z));
                    velc = getVelocity(openvdb::Coord(x, y, z), velocity);
                    velp = getVelocity(openvdb::Coord(x, y, z), velBeforeUpdate);

                    double mass = faccessor.getValue(openvdb::Coord(x, y, z));
                    // if (mass > 0.1)
                    // {
                        double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);//should it be mass*?
                        // cw = mass;
                        weight += cw;
                        
                        // cw = 1.0;
                        // weight = 1.0;//experi
                        // The difference is not between current particle velocity and grid velocity, but it's between previous and current grid velocities
                        delta += (velc - velp) * cw;
                    // }
                }
            }
        }
    }
    if (weight == 0)
    {
        return openvdb::Vec3d(0, 0, 0);
    }
    return delta/weight;
    // return delta;
}

void p2gCatmullRom(openvdb::Vec3d c, openvdb::Vec3d velc, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr weights, int bound, std::mutex ***locks, openvdb::FloatGrid::Ptr solidGrid)
{
    int fcx = round(c.x());
    int fcy = round(c.y());
    int fcz = round(c.z());

    int minx = fcx - 1 > -1 * bound ? fcx - 1 : -1 * bound;
    int miny = fcy - 1 > -1 * bound ? fcy - 1 : -1 * bound;
    int minz = fcz - 1 > -1 * bound ? fcz - 1 : -1 * bound;
    int maxx = fcx + 1 < bound ? fcx + 1 : bound;
    int maxy = fcy + 1 < bound ? fcy + 1 : bound;
    int maxz = fcz + 1 < bound ? fcz + 1 : bound;

    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();

    for (int x = minx; x <= maxx; ++x)
    {
        for (int y = miny; y <= maxy; ++y)
        {
            for (int z = minz; z <= maxz; ++z)
            {
                // std::mutex mu = ;
                // {
                    // tbb::M::scoped_lock myLock(locks[x+bound][y+bound][z+bound]);
                openvdb::Coord tc = openvdb::Coord(x, y, z);
                if (!isSolid(tc, solidGrid) && isWithinBounds(tc, bound - 2))
                {
                    locks[x+bound][y+bound][z+bound].lock();
                    double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                    waccessor.setValue(tc, waccessor.getValue(tc) + cw);
                    // std::cout << tc << " " << waccessor.getValue(tc) << std::endl;
                    vaccessor.setValue(tc, vaccessor.getValue(tc) + cw * velc);
                    locks[x+bound][y+bound][z+bound].unlock();
                }
                // }
            }
        }
    }
}




void setA(double dx, double dt, double rho, openvdb::FloatGrid::Ptr grid, openvdb::FloatGrid::Ptr Adiag, openvdb::FloatGrid::Ptr Aplusi, openvdb::FloatGrid::Ptr Aplusj, openvdb::FloatGrid::Ptr Aplusk, openvdb::FloatGrid::Ptr solidGrid)
{
    double scale = dt/(rho * dx * dx);
    openvdb::FloatGrid::Accessor daccessor = Adiag->getAccessor();
    openvdb::FloatGrid::Accessor iaccessor = Aplusi->getAccessor();
    openvdb::FloatGrid::Accessor jaccessor = Aplusj->getAccessor();
    openvdb::FloatGrid::Accessor kaccessor = Aplusk->getAccessor();
    openvdb::FloatGrid::Accessor gaccessor = grid->getAccessor();

    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    double val = gaccessor.getValue(c);        
                    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);


                    if (!isSolid(c, solidGrid) && val > 0)
                    {
                            if (!isSolid(ipjk, solidGrid))
                            {
                                double val2 = gaccessor.getValue(ipjk);
                                if (val2 > 0)
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                    daccessor.setValue(ipjk, daccessor.getValue(ipjk) + scale);
                                    iaccessor.setValue(c, -1 * scale);
                                }
                                else
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                }
                            }

                            if (!isSolid(ijpk, solidGrid))
                            {
                                double val2 = gaccessor.getValue(ijpk);
                                if (val2 > 0)
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                    daccessor.setValue(ijpk, daccessor.getValue(ijpk) + scale);
                                    jaccessor.setValue(c, -1 * scale);
                                }
                                else
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                }
                            }


                            if (!isSolid(ijkp, solidGrid))
                            {
                                double val2 = gaccessor.getValue(ijkp);
                                if (val2 > 0)
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                    daccessor.setValue(ijkp, daccessor.getValue(ijkp) + scale);
                                    kaccessor.setValue(c, -1 * scale);
                                }
                                else
                                {
                                    daccessor.setValue(c, daccessor.getValue(c) + scale);
                                }
                            }
                    }
                    else
                    {
                        if (!isSolid(c, solidGrid) && isWithinBounds(c, 13))
                        {
                                if (!isSolid(ipjk, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ipjk);
                                    if (val2 > 0)
                                    {
                                        // daccessor.setValue(c, daccessor.getValue(c) + scale);
                                        daccessor.setValue(ipjk, daccessor.getValue(ipjk) + scale);
                                        // iaccessor.setValue(c, -1 * scale);
                                    }

                                }

                                if (!isSolid(ijpk, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ijpk);
                                    if (val2 > 0)
                                    {
                                        // daccessor.setValue(c, daccessor.getValue(c) + scale);
                                        daccessor.setValue(ijpk, daccessor.getValue(ijpk) + scale);
                                        // jaccessor.setValue(c, -1 * scale);
                                    }
                                }


                                if (!isSolid(ijkp, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ijkp);
                                    if (val2 > 0)
                                    {
                                        // daccessor.setValue(c, daccessor.getValue(c) + scale);
                                        daccessor.setValue(ijkp, daccessor.getValue(ijkp) + scale);
                                        // kaccessor.setValue(c, -1 * scale);
                                    }
                                }
                            }
                    }
                }
            }
    }

}

void setRHS(double dx, openvdb::FloatGrid::Ptr grid, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr rhs, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3d gravity, double dt)
{
    double scale = 1.0/dx;
    openvdb::FloatGrid::Accessor raccessor = rhs->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor gaccessor = grid->getAccessor();
    openvdb::Vec3d g = gravity*dt;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    double val = gaccessor.getValue(c);

                    // It may in future create segmentation fault as it is not within boundary
                    openvdb::Coord imjk(c.x()-1, c.y(), c.z());
                    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());

                    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                    openvdb::Coord ijmk(c.x(), c.y()-1, c.z());

                    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
                    openvdb::Coord ijkm(c.x(), c.y(), c.z()-1);

                    openvdb::Vec3d v = vaccessor.getValue(c);
                    openvdb::Vec3d vi = vaccessor.getValue(ipjk);
                    openvdb::Vec3d vj = vaccessor.getValue(ijpk);
                    openvdb::Vec3d vk = vaccessor.getValue(ijkp);

                    if (val > 0 && !isSolid(c, solidGrid))
                    {
                        if (isWithinBounds(imjk, 15) && isSolid(imjk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.x() + g.x())));
                        }
                        if (isWithinBounds(ipjk, 15) && isSolid(ipjk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) + (scale * (vi.x() + g.x())));
                        }

                        if (isWithinBounds(ijmk, 15) && isSolid(ijmk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.y() + g.y())));
                        }
                        if (isWithinBounds(ijpk, 15) && isSolid(ijpk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) + (scale * (vj.y() + g.y())));
                        }

                        if (isWithinBounds(ijkm, 15) && isSolid(ijkm, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.z() + g.z())));
                        }
                        if (isWithinBounds(ijkp, 15) && isSolid(ijkp, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) + (scale * (vk.z() + g.z())));
                        }

                    }
                }
            }
    }

}

void setA2(openvdb::Int32Grid::Ptr indices, Eigen::VectorXd &x, Eigen::SparseMatrix<double> &A, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr Adiag, openvdb::FloatGrid::Ptr Aplusi, openvdb::FloatGrid::Ptr Aplusj, openvdb::FloatGrid::Ptr Aplusk)
{
    std::vector<Triplet> tripletList;
    openvdb::FloatGrid::Accessor iaccessor = Aplusi->getAccessor();
    openvdb::FloatGrid::Accessor jaccessor = Aplusj->getAccessor();
    openvdb::FloatGrid::Accessor kaccessor = Aplusk->getAccessor();
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    openvdb::FloatGrid::Accessor diaccessor = Adiag->getAccessor();

    for(int xi = -15; xi <=15; xi++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(xi, y, z);
                    if (diaccessor.getValue(c) != 0)
                    {
                        int index = indaccessor.getValue(c);
                        openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                        openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                        openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

                        if (iaccessor.getValue(c) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(ipjk), iaccessor.getValue(c)));
                        }
                        if (jaccessor.getValue(c) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(ijpk), jaccessor.getValue(c)));
                        }
                        if (kaccessor.getValue(c) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(ijkp), kaccessor.getValue(c)));
                        }

                        openvdb::Coord imjk(c.x()-1, c.y(), c.z());
                        openvdb::Coord ijmk(c.x(), c.y()-1, c.z());
                        openvdb::Coord ijkm(c.x(), c.y(), c.z()-1);
                        if (iaccessor.getValue(imjk) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(imjk), iaccessor.getValue(imjk)));
                        }
                        if (jaccessor.getValue(ijmk) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(ijmk), jaccessor.getValue(ijmk)));
                        }
                        if (kaccessor.getValue(ijkm) != 0)
                        {
                            tripletList.push_back(Triplet(index, indaccessor.getValue(ijkm), kaccessor.getValue(ijkm)));
                        }

                        tripletList.push_back(Triplet(index, index, diaccessor.getValue(c)));
                        x(index) = daccessor.getValue(c);
                    }
                }
            }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

void setOnlyB(openvdb::Int32Grid::Ptr indices, Eigen::VectorXd &x, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr Adiag)
{
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    openvdb::FloatGrid::Accessor adaccessor = Adiag->getAccessor();

    for(int xi = -15; xi <=15; xi++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(xi, y, z);
                    if (adaccessor.getValue(c) != 0)
                    {
                        int index = indaccessor.getValue(c);
                        x(index) = daccessor.getValue(c);
                    }
                }
            }
    }
}

void setDiver(double dx, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr diver, openvdb::FloatGrid::Ptr rhs, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::FloatGrid::Accessor daccessor = diver->getAccessor();
    openvdb::FloatGrid::Accessor raccessor = rhs->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (faccessor.getValue(c) >  0 && !isSolid(c, solidGrid))
                    {
                        openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                        openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                        openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

                        openvdb::Vec3d v = vaccessor.getValue(c);
                        openvdb::Vec3d vi = vaccessor.getValue(ipjk);
                        openvdb::Vec3d vj = vaccessor.getValue(ijpk);
                        openvdb::Vec3d vk = vaccessor.getValue(ijkp);
                        double u = 0;
                        double vd = 0;
                        double w = 0;
                        if (!isSolid(ipjk, solidGrid))
                        {
                            u = (vi.x() - v.x())/dx;
                        }
                        if (!isSolid(ijpk, solidGrid))
                        {
                            vd = (vj.y() - v.y())/dx;
                        }
                        if (!isSolid(ijkp, solidGrid))
                        {
                            w = (vk.z() - v.z())/dx;
                        }

                        daccessor.setValue(c, (raccessor.getValue(c)) - u - vd - w);
                    }
                }
            }
    }
}

void velUpdate(openvdb::Int32Grid::Ptr indices, double dx, double dt, double rho, openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr fluidGrid, Eigen::VectorXd pressure, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3d gravity, openvdb::BoolGrid::Ptr defined)
{
    double scale = dt/(rho * dx);
    openvdb::FloatGrid::Accessor faccessor = fluidGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);

                    // It may in future lead to segmentation fault
                    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);

                    double val = faccessor.getValue(c);
                    if (!isSolid(c, solidGrid))
                    {
                        if (val > 0)
                        {
                            double pre = pressure(indaccessor.getValue(c));
                            openvdb::Vec3d g = gravity*dt;
                            double u = vaccessor.getValue(c).x() - scale * pre + g.x();
                            double v = vaccessor.getValue(c).y() - scale * pre + g.y();
                            double w = vaccessor.getValue(c).z() - scale * pre + g.z();
                            vaccessor.setValue(c, openvdb::Vec3d(u, v, w));
                            daccessor.setValue(c, true);
                            if (isWithinBounds(ipjk, 15))
                            {
                                double up = vaccessor.getValue(ipjk).x() + scale * pre;
                                vaccessor.setValue(ipjk, openvdb::Vec3d(up, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
                                daccessor.setValue(ipjk, true);
                            }

                            if (isWithinBounds(ijpk, 15))
                            {
                                double vp = vaccessor.getValue(ijpk).y() + scale * pre;
                                vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), vp, vaccessor.getValue(ijpk).z()));
                                daccessor.setValue(ijpk, true);
                            }

                            if (isWithinBounds(ijkp, 15))
                            {
                                double wp = vaccessor.getValue(ijkp).z() + scale * pre;
                                vaccessor.setValue(ijkp, openvdb::Vec3d(vaccessor.getValue(ijkp).x(), vaccessor.getValue(ijkp).y(), wp));
                                daccessor.setValue(ijkp, true);
                            }
                        }
                    }

                }
            }
        }
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
                    if (isSolid(c, solidGrid))
                    {
                        vaccessor.setValue(c, openvdb::Vec3d(0, 0, 0));
                        daccessor.setValue(c, true);
                        if (isWithinBounds(ipjk, 15))
                        {
                            vaccessor.setValue(ipjk, openvdb::Vec3d(0, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
                            daccessor.setValue(ipjk, true);
                        }
                        if (isWithinBounds(ijpk, 15))
                        {
                            vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), 0, vaccessor.getValue(ijpk).z()));
                            daccessor.setValue(ijpk, true);
                        }
                        if (isWithinBounds(ijkp, 15))
                        {
                            vaccessor.setValue(ijkp, openvdb::Vec3d(vaccessor.getValue(ijkp).x(), vaccessor.getValue(ijkp).y(), 0));
                            daccessor.setValue(ijkp, true);
                        }
                    }
                }
            }
    }
}

void extrapolate(openvdb::Vec3dGrid::Ptr velocity, openvdb::BoolGrid::Ptr defined, int bound, openvdb::FloatGrid::Ptr solidGrid)
{
    openvdb::Int32Grid::Ptr numTimesDefined = openvdb::Int32Grid::create();
    openvdb::Int32Grid::Accessor ndaccessor = numTimesDefined->getAccessor();
    numTimesDefined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * bound), openvdb::Coord(bound)), 0);
    numTimesDefined->tree().voxelizeActiveTiles();
    openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
    std::vector<openvdb::Coord> definedCoord;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (daccessor.getValue(c) && !isSolid(c, solidGrid) && isWithinBounds(c, 13))
                    {
                        int minx = c.x() == -1*bound ? c.x() : c.x() - 1;
                        int miny = c.y() == -1*bound ? c.y() : c.y() - 1;
                        int minz = c.z() == -1*bound ? c.z() : c.z() - 1;

                        int maxx = c.x() == bound ? c.x() : c.x() + 1;
                        int maxy = c.y() == bound ? c.y() : c.y() + 1;
                        int maxz = c.z() == bound ? c.z() : c.z() + 1;

                        for (int i = minx; i <= maxx; ++i)
                        {
                            for (int j = miny; j <= maxy; ++j)
                            {
                                for (int k = minz; k <= maxz; ++k)
                                {
                                    openvdb::Coord neighbour = openvdb::Coord(i, j, k);
                                    if (!daccessor.getValue(neighbour))
                                    {
                                        if (ndaccessor.getValue(neighbour) == 0)
                                        {
                                            definedCoord.push_back(neighbour);
                                        }
                                        ndaccessor.setValue(neighbour, ndaccessor.getValue(neighbour) + 1);
                                        vaccessor.setValue(neighbour, vaccessor.getValue(c) + vaccessor.getValue(neighbour));
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    for (int i = 0; i < definedCoord.size(); ++i)
    {
        vaccessor.setValue(definedCoord[i], vaccessor.getValue(definedCoord[i])/ndaccessor.getValue(definedCoord[i]));
        daccessor.setValue(definedCoord[i], true);
    }
    while(definedCoord.size()) 
    {
        std::vector<openvdb::Coord> tempCoord;
        for (int iter = 0; iter < definedCoord.size(); ++iter)
        {
            openvdb::Coord c = definedCoord[iter];
            int minx = c.x() == -1*bound ? c.x() : c.x() - 1;
            int miny = c.y() == -1*bound ? c.y() : c.y() - 1;
            int minz = c.z() == -1*bound ? c.z() : c.z() - 1;

            int maxx = c.x() == bound ? c.x() : c.x() + 1;
            int maxy = c.y() == bound ? c.y() : c.y() + 1;
            int maxz = c.z() == bound ? c.z() : c.z() + 1;

            for (int i = minx; i <= maxx; ++i)
            {
                for (int j = miny; j <= maxy; ++j)
                {
                    for (int k = minz; k <= maxz; ++k)
                    {
                        openvdb::Coord neighbour = openvdb::Coord(i, j, k);
                        if (!daccessor.getValue(neighbour))
                        {
                            if (ndaccessor.getValue(neighbour) == 0)
                            {
                                tempCoord.push_back(neighbour);
                            }
                            ndaccessor.setValue(neighbour, ndaccessor.getValue(neighbour) + 1);
                            vaccessor.setValue(neighbour, vaccessor.getValue(c) + vaccessor.getValue(neighbour));
                        }
                    }
                }
            }
        }
        for (int i = 0; i < tempCoord.size(); ++i)
        {
            vaccessor.setValue(tempCoord[i], vaccessor.getValue(tempCoord[i])/ndaccessor.getValue(tempCoord[i]));
            daccessor.setValue(tempCoord[i], true);
        }
        definedCoord.clear();
        definedCoord = tempCoord;
    } 
}

struct PointList
{
    std::vector<openvdb::Vec3d> positions;
    std::vector<openvdb::Vec3d> velocities;
    std::vector<Eigen::Matrix3d> FEvec;
    std::vector<Eigen::Matrix3d> FPvec;
    std::vector<Eigen::Matrix3d> gradV;
    std::vector<double> volume;
    std::vector<bool> initializedVelocities;
    openvdb::FloatGrid::Ptr weights;
    openvdb::BoolGrid::Ptr defined;
    std::mutex ***locks;
    std::map<std::pair<int, int>, Eigen::Matrix3d> mapMatrix;
    // Eigen::Matrix3d ***dPsydx2;
    int boundary;
    PointList(){};

    void populateMatrices(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &b, double beta, double dt, int numActive, openvdb::Int32Grid::Ptr indices, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3dGrid::Ptr gridForces, openvdb::FloatGrid::Ptr fluid, openvdb::Vec3dGrid::Ptr velocity, openvdb::Vec3d gravity)
    {
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
        std::vector<Triplet> tripletList;
        double maxForceCoeff = 0;
        double maxForceCoeff2 = 0;
        double maxMi = 0;
        bool yes = false;
        openvdb::Vec3d Force;
        // Again hardcoded according to boundary
        for (int x = -15; x <= 15; ++x)
        {
            for (int y = -15; y <= 15; ++y)
            {
                for (int z = -15; z <= 15; ++z)
                {
                    openvdb::Coord xi(x, y, z);
                    double mi = faccessor.getValue(xi);
                    int k = indaccessor.getValue(xi);
                    if (mi > 0.1)
                    {
                        yes = true;
                        openvdb::Vec3d v = vaccessor.getValue(xi);
                        openvdb::Vec3d f = gaccessor.getValue(xi);
                        double maxf = std::max(fabs(f.x()), std::max(fabs(f.y()), fabs(f.z())));
                        if (maxf > maxForceCoeff)
                        {
                            maxForceCoeff = maxf;
                        }
                        if (maxf/mi > maxForceCoeff2)
                        {
                            maxForceCoeff2 = maxf/mi;
                            maxMi = mi;
                            Force = dt*f/mi;
                        }

                        Eigen::Vector3d vs(v.x() + dt*((1.0/mi)*f.x() + gravity.x()), v.y() + dt*((1.0/mi)*f.y() + gravity.y()), v.z() + dt*((1.0/mi)*f.z()+gravity.z()));
                        b[k*3] = vs(0);
                        b[k*3+1] = vs(1);
                        b[k*3+2] = vs(2);
                    }
                }
            }
        }
        std::cout << "Max Force " << Force << " " << maxMi << " " << maxForceCoeff2 << " " << yes << std::endl;
        // std::cout << b << std::endl;
        for(std::map<std::pair<int, int>, Eigen::Matrix3d>::iterator it = mapMatrix.begin(); it != mapMatrix.end(); it++)
        {
            Eigen::Matrix3d mt;
            int i = it->first.first;
            int j = it->first.second;
            // std::cout << it->second << std::endl;
            if (i == j)
            {
                mt = Eigen::Matrix3d::Identity() + beta*dt*dt*(it->second);
            }
            else
            {
                mt = beta*dt*dt*(it->second);
            }
            for (int p = 0; p < 3; ++p)
            {
                for (int q = 0; q < 3; ++q)
                {
                    tripletList.push_back(Triplet(i*3+p, j*3+q, mt(p,q)));
                }
            }

        }
        std::cout << "GARR" << std::endl;
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        std::cout << "after" << std::endl;

    }
    void initialize(int bound, openvdb::math::Transform::Ptr trans)
    {
        weights = openvdb::FloatGrid::create();
        weights->setTransform(trans);
        boundary = bound;
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        weights->tree().voxelizeActiveTiles();

        defined = openvdb::BoolGrid::create();
        defined->setTransform(trans);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        defined->tree().voxelizeActiveTiles();

        locks = new std::mutex**[2*boundary+1];
        for (int i = 0; i < (2*boundary+1); ++i)
        {
            locks[i] = new std::mutex*[2*boundary+1];
            for (int j = 0; j < (2*boundary+1); ++j)
            {
                locks[i][j] = new std::mutex[(2*boundary+1)*(2*boundary+1)];
            }
        }
        // std::cout << "HERE" << std::endl;

        // dPsydx2 = new Eigen::Matrix3d**[(2*boundary+1)*(2*boundary+1)*(2*boundary+1)];
        // for (int i = 0; i < (2*boundary+1)*(2*boundary+1)*(2*boundary+1); ++i)
        // {
        //     dPsydx2[i] = new Eigen::Matrix3d*[(2*boundary+1)*(2*boundary+1)*(2*boundary+1)];
        //     for (int j = 0; j < (2*boundary+1)*(2*boundary+1)*(2*boundary+1); ++j)
        //     {
        //         dPsydx2[i][j] = NULL;
        //     }            
        // }
        // std::cout << "There" << std::endl;
    }
    openvdb::Index64 size() const { return openvdb::Index64(positions.size()); }

    // Bad approach used here
    void add(const openvdb::Vec3d &p) 
    { 
        if(fabs(p.x()) < boundary - 2 && fabs(p.y()) < boundary - 2 && fabs(p.z()) < boundary - 2) 
        {
            positions.push_back(p); 
            // if (p.y() > -11)
            // {
            //     velocities.push_back(openvdb::Vec3d(0, 20, 0)); 
            // }
            // else
            // {
            //     velocities.push_back(openvdb::Vec3d(0, 0, 0)); 
            // }
            velocities.push_back(openvdb::Vec3d(0, -50, 0)); 
            initializedVelocities.push_back(false); 
            FEvec.push_back(Eigen::Matrix3d::Identity()); 
            FPvec.push_back(Eigen::Matrix3d::Identity());
            gradV.push_back(Eigen::Matrix3d::Zero());
            volume.push_back(0.0);
        }
    }
    // Right now threading is used here. But I believe better approach will be to use K-D trees
    // double interpolate(openvdb::Vec3d xyz, double &sum, std::mutex &l)
    // {
    //     double tsum = 0;
    //     tbb::parallel_for( size_t(0), size(), [&]( size_t i )
    //     {
    //         tsum = spline(positions[i].x() - xyz.x()) * spline(positions[i].y() - xyz.y()) * spline(positions[i].z() - xyz.z());
    //         if (tsum != 0)
    //         {
    //             l.lock();
    //             sum += tsum;
    //             l.unlock();
    //         }
    //     }
    //     );

    //     return sum;
    // }
    void updateDeformationGradient(openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr solidGrid, double dt, double dx, double thetac, double thetas)
    {
        double minv = 1 - thetac;
        double maxv = 1 + thetas;
        // for (int i = 0; i < size(); ++i)
        // {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());
            openvdb::Coord fv(fcx, fcy, fcz);
            openvdb::Vec3d fvv(fcx, fcy, fcz);
            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;
            gradV[i] = Eigen::Matrix3d::Zero();
            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            // Eigen::Vector3d gradSpline = getGradW(tc, c);
                            Eigen::Vector3d gradSpline = getGradW(tc, c);
                            openvdb::Vec3d vel = vaccessor.getValue(tc);
                            Eigen::Vector3d evel(vel.x(), vel.y(), vel.z());
                            // std::cout << tc << std::endl;
                            // std::cout << fvv << std::endl;
                            // std::cout << gradSpline.transpose() << std::endl;
                            // std::cout << std::endl;
                            // std::cout << tc << std::endl;
                            Eigen::Matrix3d gradient = evel * gradSpline.transpose();
                            // Eigen::Matrix3d gradient = getGradV(tc, p, dx, velocity);
                            gradV[i] += gradient;
                        }
                    }
                }
            }
            // std::cout << "DONE" << std::endl;

        });

        // }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
        // for (int i = 0; i < size(); ++i)
        // {
            // std::cout << gradV[i] << " " << positions[i] << std::endl;
            Eigen::Matrix3d tFE = (Eigen::Matrix3d::Identity() + dt * gradV[i]) * FEvec[i];
            Eigen::Matrix3d F =  tFE * FPvec[i];
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(tFE, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Eigen::Matrix3d singular = svd.singularValues();
            Eigen::Vector3d singular = svd.singularValues();
            singular(0) = singular(0) > minv ? singular(0) : minv;
            singular(0) = singular(0) < maxv ? singular(0) : maxv;
            singular(1) = singular(1) > minv ? singular(1) : minv;
            singular(1) = singular(1) < maxv ? singular(1) : maxv;
            singular(2) = singular(2) > minv ? singular(2) : minv;
            singular(2) = singular(2) < maxv ? singular(2) : maxv;
            FEvec[i] = svd.matrixU() * singular.asDiagonal() * svd.matrixV().transpose();
            FPvec[i] = svd.matrixV() * singular.asDiagonal().inverse() * svd.matrixU().transpose() * F;

        });
        // }
        double maxGrad = 0;
        double maxFe = 0;
        double maxFp = 0;
        for (int i = 0; i < size(); ++i)
        {
            double t = std::max(gradV[i].maxCoeff(), -1*gradV[i].minCoeff());
            double t2 = FPvec[i].determinant();
            double t3 = FEvec[i].determinant();
            if (t2 < 0)
            {
                std::cout << "FP determinant negative!!!" << std::endl;
            }
            if (maxGrad < t)
            {
                maxGrad = t;
            }
            if (maxFp < t2)
            {
                maxFp = t2;
            }
            if (maxFe < t3)
            {
                maxFe = t3;
            }
        }
        std::cout << "MAX " << maxGrad << " " << maxFp << " " << maxFe << std::endl;

    }

    void populateGridForces(openvdb::Vec3dGrid::Ptr gridForces, openvdb::Int32Grid::Ptr indices, openvdb::FloatGrid::Ptr solidGrid, openvdb::FloatGrid::Ptr fluid, int numActive, double mu, double lambda, double epsilon, double dx)
    {
        std::cout << "GAH" << std::endl;
        mapMatrix.clear();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();

        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
        // for (int i = 0; i < size(); ++i)
        // {
            // Eigen::JacobiSVD<Eigen::Matrix3d> svd(FEvec[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
            // std::cout << svd.matrixV() << std::endl;

            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());
            openvdb::Coord fv(fcx, fcy, fcz);
            openvdb::Vec3d fvv(fcx, fcy, fcz);

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            Eigen::Matrix3d sigma = getSigma(mu, lambda, epsilon, FEvec[i], FPvec[i]);
            // std::cout << sigma.maxCoeff() << " " << c << std::endl;
            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            double mi = faccessor.getValue(tc);
                            Eigen::Vector3d gradSpline = getGradW(tc, c);
                            // Eigen::Vector3d gradSpline = getGradW(tc, fvv);
                            // std::cout << gradSpline << std::endl;
                            Eigen::Vector3d eforce = -1*volume[i]*sigma*gradSpline;
                            // double maxef = std::max(fabs(eforce.x()), std::max())
                            openvdb::Vec3d force(eforce(0), eforce(1), eforce(2));
                            // openvdb::Vec3d wipj(0, -1.0*mi*gravity.y()*spline(tc.x() - c.x())*spline(tc.y() - c.y())*(spline(tc.z() - c.z())), 0);
                            // Eigen::Vector3d eterm2 = -1.0*mi*gravity.y()*(tc.y()+15)*gradSpline;
                            // Eigen::Vector3d eterm2 = -1.0*mi*gravity.y()*(c.y()+15)*gradSpline;
                            // openvdb::Vec3d term2(eterm2(0), eterm2(1), eterm2(2));
                            locks[x+boundary][y+boundary][z+boundary].lock();
                            // gaccessor.setValue(tc, gaccessor.getValue(tc) + force + term2);
                            gaccessor.setValue(tc, gaccessor.getValue(tc) + force);
                            locks[x+boundary][y+boundary][z+boundary].unlock();
                        }
                    }
                }
            }
            });
        // }
        // int factor = 3*numActive;
        double maxsigma = 0;
        for (int i = 0; i < size(); ++i)
        {
            // Eigen::Matrix3d sigmat = getSigma(mu, lambda, epsilon, FEvec[i], FPvec[i]);
            // maxsigma = std::max(maxsigma, sigmat.maxCoeff());
            // maxsigma = std::max(maxsigma, -1* sigmat.minCoeff());
            // std::cout << "JAAAAA " << FEvec[i] << std::endl;
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            // experiment
            // c = openvdb::Vec3d(fcx, fcy, fcz);
            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord xi = openvdb::Coord(x, y, z);
                        double mi = faccessor.getValue(xi);
                        if (!isSolid(xi, solidGrid) && mi > 0.1)
                        {
                            for (int x2 = minx; x2 <= maxx; ++x2)
                            {
                                for (int y2 = miny; y2 <= maxy; ++y2)
                                {
                                    for (int z2 = minz; z2 <= maxz; ++z2)
                                    {
                                        openvdb::Coord xj = openvdb::Coord(x2, y2, z2);
                                        if (!isSolid(xj, solidGrid) && faccessor.getValue(xj) > 0.1)
                                        {
                                            // hardcoded according to solid boundaries
                                            int indexi = indaccessor.getValue(xi);
                                            int indexj = indaccessor.getValue(xj);
                                            // std::cout << indexi << " " << indexj << " " << numActive << std::endl;
                                            // if (dPsydx2[indexi][indexj] == NULL)
                                            // {
                                            //     dPsydx2[indexi][indexj] = new Eigen::Matrix3d;
                                            //     *dPsydx2[indexi][indexj] = Eigen::Matrix3d::Zero();
                                            // }
                                            std::pair<int, int> tp = std::make_pair(indexi, indexj);
                                            if (mapMatrix.find(tp) == mapMatrix.end())
                                            {
                                                mapMatrix[tp] = Eigen::Matrix3d::Zero();
                                            }
                                            mapMatrix[tp] += (1.0/mi)*volume[i]*getdPsydx2(xi, xj, c, FEvec[i], FPvec[i], lambda, mu, epsilon);

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
                    std::cout << "DAH" << std::endl;
                    // std::cout << maxsigma << std::endl;

    }
    void updateVelocity(Eigen::VectorXd ans, openvdb::Int32Grid::Ptr indices, openvdb::Vec3dGrid::Ptr velocity, openvdb::Vec3dGrid::Ptr gridForces, double dt, openvdb::Vec3d gravity, openvdb::FloatGrid::Ptr fluid, openvdb::FloatGrid::Ptr solidGrid)
    {
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        openvdb::Vec3dGrid::Accessor gaccessor = gridForces->getAccessor();
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
        openvdb::Vec3d zero(0,0,0);
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (!isSolid(c, solidGrid))
                    {
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gaccessor.getValue(c) / faccessor.getValue(c)));
                        // std::cout << gaccessor.getValue(c) << std::endl;
                        if (faccessor.getValue(c) > 0.1)
                        {
                            int index = indaccessor.getValue(c);
                            vaccessor.setValue(c, openvdb::Vec3d(ans(3*index), ans(3*index+1), ans(3*index+2)));
                        }
                        else
                        {
                            vaccessor.setValue(c, openvdb::Vec3d(0,0,0));
                        }
                        // std::cout << gaccessor.getValue(c) << std::endl;
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gravity - gaccessor.getValue(c)));
                        // vaccessor.setValue(c, vaccessor.getValue(c) + dt * (gravity));
                    }
                }
            }
        }        
    }

    void findVolume(openvdb::FloatGrid::Ptr fluid, openvdb::FloatGrid::Ptr solidGrid)
    {
        openvdb::FloatGrid::Accessor faccessor = fluid->getAccessor();
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {
                        openvdb::Coord tc = openvdb::Coord(x, y, z);
                        if (!isSolid(tc, solidGrid))
                        {
                            volume[i] += faccessor.getValue(tc) * spline(c.x() - tc.x()) * spline(c.y() - tc.y()) * spline(c.z() - tc.z());
                        }
                    }
                }
            }
            volume[i] = 1.0/volume[i];
        });
    }
    void interpolate(openvdb::FloatGrid::Ptr containerGrid, openvdb::FloatGrid::Ptr solidGrid)
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d c = positions[i];
            int fcx = round(c.x());
            int fcy = round(c.y());
            int fcz = round(c.z());

            int minx = fcx - 1 > -1 * boundary ? fcx - 1 : -1 * boundary;
            int miny = fcy - 1 > -1 * boundary ? fcy - 1 : -1 * boundary;
            int minz = fcz - 1 > -1 * boundary ? fcz - 1 : -1 * boundary;
            int maxx = fcx + 1 < boundary ? fcx + 1 : boundary;
            int maxy = fcy + 1 < boundary ? fcy + 1 : boundary;
            int maxz = fcz + 1 < boundary ? fcz + 1 : boundary;

            openvdb::FloatGrid::Accessor caccessor = containerGrid->getAccessor();

            for (int x = minx; x <= maxx; ++x)
            {
                for (int y = miny; y <= maxy; ++y)
                {
                    for (int z = minz; z <= maxz; ++z)
                    {

                            openvdb::Coord tc = openvdb::Coord(x, y, z);
                            double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                            if (!isSolid(tc, solidGrid) && cw > 0)
                            {
                                locks[x+boundary][y+boundary][z+boundary].lock();
                                caccessor.setValue(tc, caccessor.getValue(tc) + cw);
                                locks[x+boundary][y+boundary][z+boundary].unlock();
                            }
                    }
                }
            }

        });
    }
    void interpFromGrid(openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid)
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            if (!initializedVelocities[i])
            {
                openvdb::Vec3d position = positions[i];
                    openvdb::Vec3d vel = clampedCatmullRom(position, velocity, bound, fluidGrid);
                    velocities[i] = vel;
                    initializedVelocities[i] = true;
            }
        });
    }

    void initializeAllVelocities()
    {
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            initializedVelocities[i] = true;
        });
    }
    void advect(double maxTimeStep, double dx, double &timestep, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid)
    {
        double e = 0.5;
        double maxSpeed = 0.0;
        std::mutex l;
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            

                openvdb::Vec3d vel = clampedCatmullRom(positions[i], velocity, bound, fluidGrid);
                
                // positions[i] = position;
                velocities[i] = vel;
                if (maxSpeed < velocities[i].length())
                {
                    l.lock();
                    if (maxSpeed < velocities[i].length())
                    {
                        maxSpeed = velocities[i].length();
                    }
                    l.unlock();
                }
        });
        if (maxSpeed != 0)
        {
            timestep = maxTimeStep < dx/maxSpeed ? maxTimeStep : dx/maxSpeed;
        }
        else
        {
            timestep = maxTimeStep;
        }

        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d position = positions[i];
            position += timestep * velocities[i];
            int rx = round(position.x());
            int ry = round(position.y());
            int rz = round(position.z());

            openvdb::Coord rcoord = openvdb::Coord(rx, ry, rz);
            if (isSolid(rcoord, solidGrid))
            {

                openvdb::Vec3d vx = openvdb::Vec3d(velocities[i].x() * timestep, 0, 0);
                openvdb::Vec3d vy = openvdb::Vec3d(0, velocities[i].y() * timestep, 0);
                openvdb::Vec3d vz = openvdb::Vec3d(0, 0, velocities[i].z() * timestep);

                if (isSolid(openvdb::Coord(round((positions[i]+vx).x()), positions[i].y(), positions[i].z()), solidGrid))
                {
                    velocities[i].x() *= -1.0*e;
                    // velocities[i].x() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), round((positions[i]+vy).y()), positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                    // velocities[i].y() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), round((positions[i]+vz).z())), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                    // velocities[i].z() *= 0;
                }
                // positions[i] += velocities[i]*timestep;
                positions[i] += velocities[i]*timestep;                
            }
            else
            {
                positions[i] = position;
            }
            }
        );
    }
    void FLIPadvect(double maxTimeStep, double dx, double &timestep, openvdb::Vec3dGrid::Ptr velocity, int bound, openvdb::Vec3dGrid::Ptr velBeforeUpdate, openvdb::FloatGrid::Ptr fluidGrid, openvdb::FloatGrid::Ptr solidGrid, openvdb::Vec3dGrid::Ptr normals)
    {
        double e = 0.0;
        openvdb::Vec3dGrid::Accessor naccessor = normals->getAccessor();
        double maxSpeed = 0.0;
        std::mutex l;
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d vel = CatmullRomFLIP(positions[i], velocities[i], velocity, bound, velBeforeUpdate, fluidGrid, solidGrid);
            velocities[i] += vel;
            if (maxSpeed < velocities[i].length())
            {
                l.lock();
                if (maxSpeed < velocities[i].length())
                {
                    maxSpeed = velocities[i].length();
                }
                l.unlock();
            }
        });
        if (maxSpeed != 0)
        {
            timestep = maxTimeStep < dx/maxSpeed ? maxTimeStep : dx/maxSpeed;
        }
        else
        {
            timestep = maxTimeStep;
        }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            openvdb::Vec3d position = positions[i];
            // std::cout << timestep * velocities[i] << std::endl;
            position += timestep * velocities[i];

            // Code to take care of stuck particles
            // int rx = round(position.x());
            // int ry = round(position.y());
            // int rz = round(position.z());
            int rx = position.x() > 0 ? ceil(position.x()) : floor(position.x());
            int ry = position.y() > 0 ? ceil(position.y()) : floor(position.y());
            int rz = position.z() > 0 ? ceil(position.z()) : floor(position.z());

            openvdb::Coord rcoord = openvdb::Coord(rx, ry, rz);
            if (isSolid(rcoord, solidGrid))
            {
                // if ((velocities[i].x() * (naccessor.getValue(rcoord).x())) < 0)
                // {
                //     velocities[i].x() *= -1.0 * e;
                // }
                // else
                // {
                //     if ((velocities[i].y() * (naccessor.getValue(rcoord).y())) < 0)
                //     {
                //         velocities[i].y() *= -1.0 * e;
                //     }
                //     else
                //     {
                //         if ((velocities[i].z() * (naccessor.getValue(rcoord).z())) < 0)
                //         {
                //             velocities[i].z() *= -1.0 * e;
                //         }
                //     }
                // }
                // openvdb::Vec3d vx = openvdb::Vec3d(velocities[i].x() * timestep, 0, 0);
                // openvdb::Vec3d vy = openvdb::Vec3d(0, velocities[i].y() * timestep, 0);
                // openvdb::Vec3d vz = openvdb::Vec3d(0, 0, velocities[i].z() * timestep);
                if (isSolid(openvdb::Coord(rx, positions[i].y(), positions[i].z()), solidGrid))
                {
                    velocities[i].x() *= -1.0*e;
                    // velocities[i].x() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), ry, positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                    // velocities[i].y() *= 0;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), rz), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                    // velocities[i].z() *= 0;
                }
                // positions[i] += velocities[i]*timestep;
                positions[i] += velocities[i]*timestep;                
            }
            else
            {
                positions[i] = position;
            }
            }
);
        // for (int i = 0; i < size(); ++i)
        // {
        //     int rx = round(positions[i].x());
        //     int ry = round(positions[i].y());
        //     int rz = round(positions[i].z());

        //     if (isSolid(openvdb::Coord(rx, ry, rz), solidGrid))
        //     {
        //         if ((velocities[i].x() * (rx - positions[i].x())) > 0)
        //         {
        //             velocities[i].x() *= -1.0 * e;
        //         }
        //         if ((velocities[i].y() * (ry - positions[i].y())) > 0)
        //         {
        //             velocities[i].y() *= -1.0 * e;
        //         }
        //         if ((velocities[i].z() * (rz - positions[i].z())) > 0)
        //         {
        //             velocities[i].z() *= -1.0 * e;
        //         }
        //     }
        // }

    }
    void out(openvdb::FloatGrid::Ptr outgrid)
    {
        openvdb::FloatGrid::Accessor outGridAccessor = outgrid->getAccessor();
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            int rx = round(positions[i].x());
            int ry = round(positions[i].y());
            int rz = round(positions[i].z());
            if (rx < 50)
            {
                outGridAccessor.setValue(openvdb::Coord(rx, ry, rz), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx+1, ry, rz), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx, ry+1, rz), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx, ry, rz+1), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx, ry, rz-1), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx, ry-1, rz), 1);
                // outGridAccessor.setValue(openvdb::Coord(rx-1, ry, rz), 1);
            }
        }); 
    }
        void resample(int numParticlesPerCell)
        {
            openvdb::Int32Grid::Ptr number = openvdb::Int32Grid::create();
            number->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
            openvdb::Int32Grid::Accessor naccessor = number->getAccessor();
            std::vector<int> particlesToRemove;
            tbb::parallel_for( size_t(0), size(), [&]( size_t i )
            {
                int rx = round(positions[i].x());
                int ry = round(positions[i].y());
                int rz = round(positions[i].z());
                if (rx < 50)
                {
                    openvdb::Coord rcoord = openvdb::Coord(rx, ry, rz);
                    // std::cout << rcoord << std::endl;
                    locks[rx+15][ry+15][rz+15].lock();
                    if (naccessor.getValue(rcoord) + 1 > numParticlesPerCell)
                    {
                        positions[i] = openvdb::Vec3d(100, 100, 100);
                    }
                    else
                    {
                        naccessor.setValue(rcoord, naccessor.getValue(rcoord) + 1);
                    }
                    locks[rx+15][ry+15][rz+15].unlock();
                }

            });
        }
    double addGravity(double dx, openvdb::Vec3d gravity)
    {
        std::mutex l;
        double dt = 0.5;
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            double val = (dx/(2.0 * (velocities[i] + dt * gravity).length()));
            if (val < dt)
            {
                l.lock();
                if (val < dt)
                {
                    dt = val;
                }
                l.unlock();
            }
        });
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            velocities[i] += dt*gravity;
        });

        return dt;
    }
    void P2Gtransfer(openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr solidGrid, openvdb::FloatGrid::Ptr fluid)
    {
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
                {
                    for (int z = -15; z <= 15; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        if (fabs(c.x()) > 13 || fabs(c.y()) > 13 || fabs(c.z()) > 13 || isSolid(c, solidGrid))
                        {
                            daccessor.setValue(c, true);
                        }
                    }
                }
        }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
            // if (positions[i].x() < 50)
            // {
                p2gCatmullRom(positions[i], velocities[i], velocity, weights, boundary, locks, solidGrid);
            // }
        });
        // openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();
        openvdb::FloatGrid::Accessor waccessor = fluid->getAccessor();
        // std::cout << waccessor.getValue(openvdb::Coord(-4, -4, -4)) << std::endl;
        // std::cout << waccessor.getValue(openvdb::Coord(-2, 3, -3)) << std::endl;
        // std::cout << waccessor.getValue(openvdb::Coord(-3, -2, 0)) << std::endl;
        // std::cout << waccessor.getValue(openvdb::Coord(4, 3, 2)) << std::endl;
        // std::cout << waccessor.getValue(openvdb::Coord(0, 0, 0)) << std::endl;
        for(int x = -15; x <=15; x++)
        {
            for(int y = -15; y <=15; y++)
                {
                    for (int z = -15; z <= 15; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        // std::cout << iter.getCoord() << " " << waccessor.getValue(iter.getCoord()) << std::endl;
                        double w = waccessor.getValue(c);
                        if (w > 0.1)
                        {
                            vaccessor.setValue(c, vaccessor.getValue(c)/w);
                            daccessor.setValue(c, true);
                        }
                        else
                        {
                            vaccessor.setValue(c, openvdb::Vec3d(0,0,0));
                        }
                    }
                }
        }
        // extrapolate(velocity, defined, boundary, solidGrid);
    }
};

int main(int argc, char* argv[])
{
    openvdb::initialize();
    openvdb::math::Transform::Ptr trans = openvdb::math::Transform::createLinearTransform(factor);

    // The container grid which contains the fluid
    openvdb::FloatGrid::Ptr containerGrid = openvdb::FloatGrid::create(0);
    containerGrid->setTransform(trans);
    containerGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);

    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(0);
    outputGrid->setTransform(trans);
    outputGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
    outputGrid->tree().voxelizeActiveTiles();

    openvdb::FloatGrid::Ptr solidGrid = openvdb::FloatGrid::create();
    solidGrid->setTransform(trans);
    solidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0);

    // The fluid grid represents the fluid
    openvdb::FloatGrid::Ptr fluidGrid = openvdb::FloatGrid::create(-1);
    fluidGrid->setTransform(trans);


    // For normal fluid
    // fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-13), openvdb::Coord(-10)), 0, true);

    // For pea fluid
    // fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-1), openvdb::Coord(1)), 0, true);
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -1; i <= 2; ++i)
    // {
    //     for (int j = -13; j <= -10; ++j)
    //     {
    //         for (int k = -1; k <= 2; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    // For cone
    openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    for (int i = -13; i <= 13; ++i)
    {
        for (int j = -13; j <= -10; ++j)
        {
            for (int k = -13; k <= 13; ++k)
            {
                double r = (double)(j+13)/2;
                if (i*i + k*k <= r*r)
                {
                    fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
                }
            }
        }
    }

    // For double balls
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             // double r = (double)(j+13)/2;
    //             if ((i)*(i) + k*k + (j+11)*(j+11) <= 2*2)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             // double r = (double)(j+13)/2;
    //             if ((i)*(i) + k*k + (j+7)*(j+7) <= 2*2)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For sphere
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = -13; k <= 13; ++k)
    //         {
    //             int jt = j+10;
    //             if (i*i + k*k + jt*jt <= 9)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For O
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= 13; ++j)
    //     {
    //         for (int k = 0; k <= 0; ++k)
    //         {
    //             double it = i;
    //             double jt = j+8;
    //             if (it*it + jt*jt <= 25 && it*it + jt*jt >= 16)
    //             {
    //                 fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //             }
    //         }
    //     }
    // }
    // For side fluid
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -57; i <= 57; ++i)
    // {
    //     for (int j = -57; j <= -40; ++j)
    //     {
    //         for (int k = -57; k <= -40; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    // For stable fluid
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -57; i <= 57; ++i)
    // {
    //     for (int j = -57; j <= -55; ++j)
    //     {
    //         for (int k = -57; k <= 57; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
    //         }
    //     }
    // }

    // Adiag represents the diagonal components of the sparse matrix, i.e., A[(i,j,k),(i,j,k)]
    openvdb::FloatGrid::Ptr Adiag = openvdb::FloatGrid::create(0);
    Adiag->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusj = openvdb::FloatGrid::create(0);
    Aplusj->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusi = openvdb::FloatGrid::create(0);
    Aplusi->setTransform(trans);

    openvdb::FloatGrid::Ptr Aplusk= openvdb::FloatGrid::create(0);
    Aplusk->setTransform(trans);

    openvdb::FloatGrid::Ptr diver = openvdb::FloatGrid::create(0);
    diver->setTransform(trans);

    openvdb::FloatGrid::Ptr rhs = openvdb::FloatGrid::create(0);
    rhs->setTransform(trans);

    openvdb::Int32Grid::Ptr indices = openvdb::Int32Grid::create(0);
    indices->setTransform(trans);

    // openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create();
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::Vec3dGrid::Ptr normals = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    normals->setTransform(trans);
    normals->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::FloatGrid::Accessor outGridAccessor = outputGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor normalAccessor = normals->getAccessor();
    openvdb::FloatGrid::Accessor saccessor = solidGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    openvdb::FloatGrid::Accessor caccessor = containerGrid->getAccessor();

    // Voxelize all leafs
    // openvdb::tools::activate(vels->tree(), openvdb::Vec3d(1,1,1));
    // openvdb::tools::activate(fluidGrid->tree(), 0);
    // openvdb::tools::activate(containerGrid->tree(), 0);
    // openvdb::tools::activate(solidGrid->tree(), 0);
    // openvdb::tools::activate(normals->tree(), openvdb::Vec3d(0,0,0));

    // vels->tree().voxelizeActiveTiles();
    // fluidGrid->tree().voxelizeActiveTiles();
    // containerGrid->tree().voxelizeActiveTiles();
    // solidGrid->tree().voxelizeActiveTiles();
    // normals->tree().voxelizeActiveTiles();

    using Randgen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19, 0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {

                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if (fabs(xyz.x()) > 13 || fabs(xyz.y()) > 13 || fabs(xyz.z()) > 13)
                    {
                        saccessor.setValue(xyz, 1);
                        openvdb::Vec3d vec = normalAccessor.getValue(xyz);
                        if (fabs(xyz.x()) > 13)
                        {
                            if (xyz.x() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(1, vec.y(), vec.z()));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(-1, vec.y(), vec.z()));
                            }
                        }
                        if (fabs(xyz.y()) > 13)
                        {
                            if (xyz.y() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), 1, vec.z()));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), -1, vec.z()));                    
                            }
                        }
                        if (fabs(xyz.z()) > 13)
                        {
                            if (xyz.z() < 0)
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), vec.y(), 1));
                            }
                            else
                            {
                                normalAccessor.setValue(xyz, openvdb::Vec3d(vec.x(), vec.y(), -1));                    
                            }
                        }
                    }

        // for two blocks
        // if((xyz.x() >= -11 && xyz.x() <= -6) || (xyz.x() <= 11 && xyz.x() >= 6))
        // {
        //     if((xyz.z() >= -3 && xyz.z() <= 3))
        //     {
        //         if((xyz.y() >= -13 && xyz.y() <= -8))
        //         {
        //             iter.setValue(1);
        //             outGridAccessor.setValue(xyz, 1);
        //         }
        //     }
        // }

        // for three blocks
        // if((xyz.x() >= -11 && xyz.x() <= -7) || (xyz.x() >= -2 && xyz.x() <= 2) || (xyz.x() <= 11 && xyz.x() >= 7))
        // {
        //     if((xyz.z() >= -3 && xyz.z() <= 3))
        //     {
        //         if((xyz.y() >= -13 && xyz.y() <= -8))
        //         {
        //             iter.setValue(1);
        //             outGridAccessor.setValue(xyz, 1);
        //         }
        //     }
        // }

                }
            }
    }

    // for big wall
    // for (int i = -13; i <= 13; ++i)
    // {
    //     for (int j = -13; j <= -50; ++j)
    //     {
    //         for (int k = -15; k <= -25; ++k)
    //         {
    //             openvdb::Coord xyz = openvdb::Coord(i, j, k);
    //             saccessor.setValue(xyz, 1);
    //             outGridAccessor.setValue(xyz, 1);
    //         }
    //     }
    // }
    // openvdb::GridPtrVec grids;
    PointList pos;
    pos.initialize(15, trans);
    // const openvdb::Index64 pointCount = 400;
    // Randgen mtRand;
    std::mt19937 mtRandi(0);
    openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatteri(pos, 400.f, mtRandi);
    scatteri.operator()<openvdb::FloatGrid>(*fluidGrid);

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;

    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    bool isIndiceVoxelized = false;
    bool areVoxelized = false;
    openvdb::Vec3d gravity(0, -10, 0);
    double dx = 1.0*factor;
    openvdb::Vec3dGrid::Ptr velBeforeUpdate;

    pos.initializeAllVelocities();
    openvdb::BoolGrid::Ptr defined = pos.defined -> deepCopy();
    double simulationTime = 0;
    std::string filename = "mygrids.vdb";
    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    double dt = 0.001;
    openvdb::Vec3dGrid::Ptr gridForces = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    gridForces->setTransform(trans);
for (int i = 0; i < 500; ++i)
{
    // open file
    std::string filename = "simulation/mygrids" + std::to_string(i) + ".vdb";
    openvdb::io::File file2(filename);
    openvdb::GridPtrVec grids2;
    // Randgen mtRand;
    std::mt19937 mtRand(i+1);
    // mtRand.seed(i+1);
    // openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatter(pos, 10.f, mtRand);

    
    // if (i%5 == 0)
    // {
        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
    // }
    // pos.resample(8);
    // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
    // grids.push_back(vels -> deepCopy());
    // double maxSpeed = 0;
    // for(int x = -15; x <=15; x++)
    // {
    //     for(int y = -15; y <=15; y++)
    //         {
    //             for (int z = -15; z <= 15; z++)
    //             {
    //                 openvdb::Coord xyz = openvdb::Coord(x, y, z);
    //                 double val = vaccessor.getValue(xyz).length();
    //                 maxSpeed = val < maxSpeed? maxSpeed:val;
    //             }
    //         }
    // }
    // if (maxSpeed != 0)
    // {
    //     dt = dx/maxSpeed;
    // }
    std::cout << "DT " << dt << std::endl;

    indices->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), -1, true);
    gridForces->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0,0,0));
    if (!isIndiceVoxelized)
    {
        indices->tree().voxelizeActiveTiles();
        isIndiceVoxelized = true;
    }

    int numActive = 0;
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    // if(!isSolid(xyz, solidGrid))
                    // {
                        caccessor.setValue(xyz, 0);
                    // }
                }
            }
    }



    pos.interpolate(containerGrid, solidGrid);
    pos.P2Gtransfer(vels, solidGrid, containerGrid);
    if (i == 0)
    {
        pos.findVolume(containerGrid, solidGrid);
    }
    // for(int x = -15; x <=15; x++)
    // {
    //     for(int y = -15; y <=15; y++)
    //         {
    //             for (int z = -15; z <= 15; z++)
    //             {
    //                 openvdb::Coord xyz = openvdb::Coord(x, y, z);
    //                 if(!isSolid(xyz, solidGrid) && caccessor.getValue(xyz) > 0)
    //                 {
    //                     vaccessor.setValue(xyz, vaccessor.getValue(xyz) + dt * gravity);
    //                 }
    //             }
    //         }
    // }

    // for(int x = -15; x <=15; x++)
    // {
    //     for(int y = -15; y <=15; y++)
    //         {
    //             for (int z = -15; z <= 15; z++)
    //             {
    //                 openvdb::Coord xyz = openvdb::Coord(x, y, z);
    //                 if(!isSolid(xyz, solidGrid))
    //                 {
    //                     vaccessor.setValue(xyz, vaccessor.getValue(xyz) + dt * gravity);
    //                 }
    //             }
    //         }
    // }

    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if(!isSolid(xyz, solidGrid) && isWithinBounds(xyz, 13))
                    {
                        if (caccessor.getValue(xyz) > 0.1)
                        {
                            indaccessor.setValue(xyz, numActive);
                            numActive++;
                        }
                    }
                }
            }
    }
    for(int x = -15; x <=15; x++)
    {
        for(int y = -15; y <=15; y++)
            {
                for (int z = -15; z <= 15; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    double val = caccessor.getValue(xyz);
                    if (!isSolid(xyz, solidGrid) && val > 0.1)
                    {
                        outGridAccessor.setValue(xyz, val);
                    }
                }
            }
    }
    Eigen::VectorXd vsolve(3*numActive);

    grids.push_back(outputGrid -> deepCopy());
    grids2.push_back(outputGrid -> deepCopy());
    // grids.push_back(vels -> deepCopy());
    // grids2.push_back(vels -> deepCopy());


    // grids.push_back(vels -> deepCopy());
    Eigen::VectorXd b2(numActive);        
    double error;
    velBeforeUpdate = vels -> deepCopy();
    double E = 48000;
    double nu = 0.47;
    double beta = 0.5;
    std::cout << "1" << std::endl;
    pos.populateGridForces(gridForces, indices, solidGrid, containerGrid, numActive, E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu)), 10, dx);
    std::cout << "2" << std::endl;
    Eigen::SparseMatrix<double> A(3*numActive, 3*numActive);    
    Eigen::VectorXd b(3*numActive);        
    pos.populateMatrices(A, b, beta, dt, numActive, indices, solidGrid, gridForces, containerGrid, vels, gravity);
    cg.compute(A);
    vsolve = cg.solve(b);
    pos.updateVelocity(vsolve, indices, vels, gridForces, dt, gravity, containerGrid, solidGrid);
    std::cout << "Error: " << cg.error() << std::endl;
    std::cout << "5" << std::endl;
    // grids.push_back(vels -> deepCopy());
    // grids2.push_back(vels -> deepCopy());
    std::cout << "3" << std::endl;
    pos.updateDeformationGradient(vels, solidGrid, dt, dx, 0.025, 0.0075);
    std::cout << "4" << std::endl;

    //     velUpdate(indices, dx, /*dt*/dt/10, /*pho*/1, vels, containerGrid, p, solidGrid, gravity, defined);

    //     diver->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
    //     rhs->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), 0, true);
    //     setRHS(dx, containerGrid, vels, rhs, solidGrid, gravity, dt);
    //     setDiver(dx, vels, diver, rhs, containerGrid, solidGrid);
    //     setOnlyB(indices, b2, diver, Adiag);
    //     // setA2(indices, b2, A, diver, Adiag, Aplusi, Aplusj, Aplusk);
    //     error = ((b - b2).norm())/(b.norm());
    // }while(error > 0.1);
    // pos.defined = defined->deepCopy();
    // std::cout << "After" << std::endl;
    // grids.push_back(vels -> deepCopy());
    // openvdb::tools::PointAdvect<openvdb::Vec3dGrid, std::vector<openvdb::Vec3d>, true, openvdb::util::NullInterrupter> pa(*vels);
    // pa.advect(pos.list, dt);
    // pos.advect(dt, vels, 15, containerGrid);
    // getUnstaggered(vels, trans);
    // maxSpeed = 0;
    // for(int x = -15; x <=15; x++)
    // {
    //     for(int y = -15; y <=15; y++)
    //         {
    //             for (int z = -15; z <= 15; z++)
    //             {
    //                 openvdb::Coord xyz = openvdb::Coord(x, y, z);
    //                 double val = vaccessor.getValue(xyz).length();
    //                 maxSpeed = val < maxSpeed? maxSpeed:val;
    //             }
    //         }
    // }
    // if (maxSpeed != 0)
    // {
    //     dt = dt < dx/maxSpeed ? dt : dx/maxSpeed;
    // }

    // pos.advect(0.1, dx, dt, vels, 15, containerGrid, solidGrid);
    // Same to do for pic
    pos.FLIPadvect(0.001, dx, dt, vels, 15, velBeforeUpdate, containerGrid, solidGrid, normals);
    std::cout << "DT " << dt << std::endl;
    if(i < 500)
    {

        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
        // // //containerGrid is dummy here
        // pos.interpFromGrid(vels, 15, containerGrid);
    }
    // std::cout << "Error:\t" << error << std::endl;
    std::cout << "Iteration:\t" << i+1 << std::endl;
    simulationTime += dt;
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(15)), openvdb::Vec3d(0, 0, 0), true);
    // std::cout << "Time delta:\t" << simulationTime << std::endl;
    // file.write(grids);
    file2.write(grids2);
    file2.close();

}
    file.write(grids);
    file.close();

end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
std::cout << "Time Taken " << cpu_time_used/15 << " minutes" << std::endl; 
}
