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
#include <Eigen/IterativeLinearSolvers>
#include <mutex>
#include <time.h>

clock_t end;
double cpu_time_used;
clock_t start = clock();
typedef Eigen::Triplet<double> Triplet;
double spline(double x)
{
    if (x < 0)
    {
        x *= -1.0;
    }
    if (x < 0.5)
    {
        return 1.5 * (4.0*x*x*x - 4.0*x*x + 2.0/3.0);
    }
    if (x < 1.0)
    {
        return 1.5* ((-8.0 * (x * x * x)/6.0) + 4.0*x*x - 4.0*x + 4.0/3.0);
    }
    return 0;
}
bool isWithinBounds(openvdb::Coord xyz, int bound)
{
    if (abs(xyz.x()) > bound || abs(xyz.y()) > bound || abs(xyz.z()) > bound)
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
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), openvdb::Vec3d(0, 0, 0));
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    openvdb::Vec3dGrid::Accessor ovaccessor = original->getAccessor();
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
        {
            for (int z = -60; z <= 60; z++)
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
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), openvdb::Vec3d(0, 0, 0));
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
        {
            for (int z = -60; z <= 60; z++)
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
                if (isWithinBounds(openvdb::Coord(x, y, z), 58))
                {
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
                if(isWithinBounds(openvdb::Coord(x, y, z), 58))
                {
                    velc = getVelocity(openvdb::Coord(x, y, z), velocity);
                    velp = getVelocity(openvdb::Coord(x, y, z), velBeforeUpdate);

                    // double mass = faccessor.getValue(openvdb::Coord(x, y, z));
                    // if (mass > 0)
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
                openvdb::Coord tc = openvdb::Coord(x, y, z);
                if (!isSolid(tc, solidGrid) && isWithinBounds(tc, bound - 2))
                {
                    locks[x+bound][y+bound][z+bound].lock();
                    double cw = spline(c.x() - x) * spline(c.y() - y) * spline(c.z() - z);
                    waccessor.setValue(tc, waccessor.getValue(tc) + cw);
                    vaccessor.setValue(tc, vaccessor.getValue(tc) + cw * velc);
                    locks[x+bound][y+bound][z+bound].unlock();
                }
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

    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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
                        if (!isSolid(c, solidGrid) && isWithinBounds(c, 58))
                        {
                                if (!isSolid(ipjk, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ipjk);
                                    if (val2 > 0)
                                    {
                                        daccessor.setValue(ipjk, daccessor.getValue(ipjk) + scale);
                                    }

                                }

                                if (!isSolid(ijpk, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ijpk);
                                    if (val2 > 0)
                                    {
                                        daccessor.setValue(ijpk, daccessor.getValue(ijpk) + scale);
                                    }
                                }


                                if (!isSolid(ijkp, solidGrid))
                                {
                                    double val2 = gaccessor.getValue(ijkp);
                                    if (val2 > 0)
                                    {
                                        daccessor.setValue(ijkp, daccessor.getValue(ijkp) + scale);
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
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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
                        if (isWithinBounds(imjk, 60) && isSolid(imjk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.x() + g.x())));
                        }
                        if (isWithinBounds(ipjk, 60) && isSolid(ipjk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) + (scale * (vi.x() + g.x())));
                        }

                        if (isWithinBounds(ijmk, 60) && isSolid(ijmk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.y() + g.y())));
                        }
                        if (isWithinBounds(ijpk, 60) && isSolid(ijpk, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) + (scale * (vj.y() + g.y())));
                        }

                        if (isWithinBounds(ijkm, 60) && isSolid(ijkm, solidGrid))
                        {
                            raccessor.setValue(c, raccessor.getValue(c) - (scale * (v.z() + g.z())));
                        }
                        if (isWithinBounds(ijkp, 60) && isSolid(ijkp, solidGrid))
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

    for(int xi = -60; xi <=60; xi++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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

    for(int xi = -60; xi <=60; xi++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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
                            if (isWithinBounds(ipjk, 60))
                            {
                                double up = vaccessor.getValue(ipjk).x() + scale * pre;
                                vaccessor.setValue(ipjk, openvdb::Vec3d(up, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
                                daccessor.setValue(ipjk, true);
                            }

                            if (isWithinBounds(ijpk, 60))
                            {
                                double vp = vaccessor.getValue(ijpk).y() + scale * pre;
                                vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), vp, vaccessor.getValue(ijpk).z()));
                                daccessor.setValue(ijpk, true);
                            }

                            if (isWithinBounds(ijkp, 60))
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
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    openvdb::Coord ipjk(c.x()+1, c.y(), c.z());
                    openvdb::Coord ijpk(c.x(), c.y()+1, c.z());
                    openvdb::Coord ijkp(c.x(), c.y(), c.z()+1);
                    if (isSolid(c, solidGrid))
                    {
                        vaccessor.setValue(c, openvdb::Vec3d(0, 0, 0));
                        daccessor.setValue(c, true);
                        if (isWithinBounds(ipjk, 60))
                        {
                            vaccessor.setValue(ipjk, openvdb::Vec3d(0, vaccessor.getValue(ipjk).y(), vaccessor.getValue(ipjk).z()));
                            daccessor.setValue(ipjk, true);
                        }
                        if (isWithinBounds(ijpk, 60))
                        {
                            vaccessor.setValue(ijpk, openvdb::Vec3d(vaccessor.getValue(ijpk).x(), 0, vaccessor.getValue(ijpk).z()));
                            daccessor.setValue(ijpk, true);
                        }
                        if (isWithinBounds(ijkp, 60))
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
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
                {
                    openvdb::Coord c = openvdb::Coord(x, y, z);
                    if (daccessor.getValue(c) && !isSolid(c, solidGrid) && isWithinBounds(c, 58))
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
    std::vector<bool> initializedVelocities;
    openvdb::FloatGrid::Ptr weights;
    openvdb::BoolGrid::Ptr defined;
    std::mutex ***locks;
    int boundary;
    PointList(){};
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
                locks[i][j] = new std::mutex[2*boundary+1];
            }
        }
    }
    openvdb::Index64 size() const { return openvdb::Index64(positions.size()); }

    // Bad approach used here
    void add(const openvdb::Vec3d &p) { if(abs(p.x()) < boundary - 2 && abs(p.y()) < boundary - 2 && abs(p.z()) < boundary - 2) {positions.push_back(p); velocities.push_back(openvdb::Vec3d(0, 0, 0)); initializedVelocities.push_back(false);}}

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

        }
        );
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
                }
                if (isSolid(openvdb::Coord(positions[i].x(), round((positions[i]+vy).y()), positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), round((positions[i]+vz).z())), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                }
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
        double e = 0;
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
            position += timestep * velocities[i];

            // Code to take care of stuck particles
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
                }
                if (isSolid(openvdb::Coord(positions[i].x(), round((positions[i]+vy).y()), positions[i].z()), solidGrid))
                {
                    velocities[i].y() *= -1.0*e;
                }
                if (isSolid(openvdb::Coord(positions[i].x(), positions[i].y(), round((positions[i]+vz).z())), solidGrid))
                {
                    velocities[i].z() *= -1.0*e;
                }
                positions[i] += velocities[i]*timestep;                
            }
            else
            {
                positions[i] = position;
            }
            }
);

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
                    locks[rx+60][ry+60][rz+60].lock();
                    if (naccessor.getValue(rcoord) + 1 > numParticlesPerCell)
                    {
                        positions[i] = openvdb::Vec3d(100, 100, 100);
                    }
                    else
                    {
                        naccessor.setValue(rcoord, naccessor.getValue(rcoord) + 1);
                    }
                    locks[rx+60][ry+60][rz+60].unlock();
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
    void P2Gtransfer(openvdb::Vec3dGrid::Ptr velocity, openvdb::FloatGrid::Ptr solidGrid)
    {
        weights->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), 0);
        defined->fill(openvdb::CoordBBox(openvdb::Coord(-1 * boundary), openvdb::Coord(boundary)), false);
        openvdb::BoolGrid::Accessor daccessor = defined->getAccessor();
        openvdb::Vec3dGrid::Accessor vaccessor = velocity->getAccessor();
        for(int x = -60; x <=60; x++)
        {
            for(int y = -60; y <=60; y++)
                {
                    for (int z = -60; z <= 60; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        if (abs(c.x()) > 58 || abs(c.y()) > 58 || abs(c.z()) > 58 || isSolid(c, solidGrid))
                        {
                            daccessor.setValue(c, true);
                        }
                    }
                }
        }
        tbb::parallel_for( size_t(0), size(), [&]( size_t i )
        {
                p2gCatmullRom(positions[i], velocities[i], velocity, weights, boundary, locks, solidGrid);
        });
        openvdb::FloatGrid::Accessor waccessor = weights->getAccessor();
        for(int x = -60; x <=60; x++)
        {
            for(int y = -60; y <=60; y++)
                {
                    for (int z = -60; z <= 60; z++)
                    {
                        openvdb::Coord c = openvdb::Coord(x, y, z);
                        double w = waccessor.getValue(c);
                        if (w > 0)
                        {
                            vaccessor.setValue(c, vaccessor.getValue(c)/w);
                            daccessor.setValue(c, true);
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
    openvdb::math::Transform::Ptr trans = openvdb::math::Transform::createLinearTransform(1);

    // The container grid which contains the fluid
    openvdb::FloatGrid::Ptr containerGrid = openvdb::FloatGrid::create(0);
    containerGrid->setTransform(trans);
    containerGrid->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);

    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(0);
    outputGrid->setTransform(trans);
    outputGrid->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
    outputGrid->tree().voxelizeActiveTiles();

    openvdb::FloatGrid::Ptr solidGrid = openvdb::FloatGrid::create();
    solidGrid->setTransform(trans);
    solidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0);

    // The fluid grid represents the fluid
    openvdb::FloatGrid::Ptr fluidGrid = openvdb::FloatGrid::create(-1);
    fluidGrid->setTransform(trans);


    // For normal fluid
    fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-20), openvdb::Coord(20)), 0, true);

    // For pea fluid
    // fluidGrid->fill(openvdb::CoordBBox(openvdb::Coord(-1), openvdb::Coord(1)), 0, true);
    // openvdb::FloatGrid::Accessor fluidAccessor = fluidGrid -> getAccessor();
    // for (int i = -1; i <= 1; ++i)
    // {
    //     for (int j = 6; j <= 9; ++j)
    //     {
    //         for (int k = -1; k <= 1; ++k)
    //         {
    //             fluidAccessor.setValue(openvdb::Coord(i, j, k), 0);
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

    openvdb::Vec3dGrid::Ptr vels = openvdb::Vec3dGrid::create();
    vels->setTransform(trans);
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::Vec3dGrid::Ptr normals = openvdb::Vec3dGrid::create(openvdb::Vec3d(0, 0, 0));
    normals->setTransform(trans);
    normals->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), openvdb::Vec3d(0, 0, 0), true);

    openvdb::FloatGrid::Accessor outGridAccessor = outputGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor normalAccessor = normals->getAccessor();
    openvdb::FloatGrid::Accessor saccessor = solidGrid->getAccessor();
    openvdb::Vec3dGrid::Accessor vaccessor = vels->getAccessor();
    openvdb::FloatGrid::Accessor caccessor = containerGrid->getAccessor();


    using Randgen = std::mersenne_twister_engine<uint32_t, 32, 351, 175, 19, 0xccab8ee7, 11, 0xffffffff, 7, 0x31b6ab00, 15, 0xffe50000, 17, 1812433253>;
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
                {

                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if (abs(xyz.x()) > 58 || abs(xyz.y()) > 58 || abs(xyz.z()) > 58)
                    {
                        saccessor.setValue(xyz, 1);
                        openvdb::Vec3d vec = normalAccessor.getValue(xyz);
                        if (abs(xyz.x()) > 58)
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
                        if (abs(xyz.y()) > 58)
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
                        if (abs(xyz.z()) > 58)
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
        //         if((xyz.y() >= -58 && xyz.y() <= -8))
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
        //         if((xyz.y() >= -58 && xyz.y() <= -8))
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
    // for (int i = -58; i <= 58; ++i)
    // {
    //     for (int j = -58; j <= -50; ++j)
    //     {
    //         for (int k = -30; k <= -25; ++k)
    //         {
    //             openvdb::Coord xyz = openvdb::Coord(i, j, k);
    //             saccessor.setValue(xyz, 1);
    //             outGridAccessor.setValue(xyz, 1);
    //         }
    //     }
    // }
    PointList pos;
    pos.initialize(60, trans);
    std::mt19937 mtRandi(0);
    openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatteri(pos, 10.f, mtRandi);
    scatteri.operator()<openvdb::FloatGrid>(*fluidGrid);

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg;

    openvdb::Int32Grid::Accessor indaccessor = indices->getAccessor();
    bool isIndiceVoxelized = false;
    bool areVoxelized = false;
    openvdb::Vec3d gravity(0, -10, 0);
    double dx = 1.0;
    openvdb::Vec3dGrid::Ptr velBeforeUpdate;

    pos.initializeAllVelocities();
    openvdb::BoolGrid::Ptr defined = pos.defined -> deepCopy();
    double simulationTime = 0;
    std::string filename = "mygrids.vdb";
    openvdb::io::File file(filename);
    openvdb::GridPtrVec grids;
    double dt = 0.1;
for (int i = 0; i < 500; ++i)
{
    // open file
    std::string filename = "simulation/mygrids" + std::to_string(i) + ".vdb";
    openvdb::io::File file2(filename);
    openvdb::GridPtrVec grids2;
    std::mt19937 mtRand(i+1);
    openvdb::tools::UniformPointScatter<PointList, std::mt19937> scatter(pos, 10.f, mtRand);

    
    vels->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), openvdb::Vec3d(0, 0, 0), true);
    // if (i%5 == 0)
    // {
        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
    // }
    std::cout << "2" << std::endl;
    pos.P2Gtransfer(vels, solidGrid);
    std::cout << "3" << std::endl;
    std::cout << "DT " << dt << std::endl;

    indices->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), -1, true);
    if (!isIndiceVoxelized)
    {
        indices->tree().voxelizeActiveTiles();
        isIndiceVoxelized = true;
    }

    int numActive = 0;
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
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


    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    if(!isSolid(xyz, solidGrid) && isWithinBounds(xyz, 58))
                    {
                        if (caccessor.getValue(xyz) > 0)
                        {
                            indaccessor.setValue(xyz, numActive);
                            numActive++;
                        }
                    }
                }
            }
    }
    for(int x = -60; x <=60; x++)
    {
        for(int y = -60; y <=60; y++)
            {
                for (int z = -60; z <= 60; z++)
                {
                    openvdb::Coord xyz = openvdb::Coord(x, y, z);
                    double val = caccessor.getValue(xyz);
                    if (!isSolid(xyz, solidGrid))
                    {
                        outGridAccessor.setValue(xyz, val);
                    }
                }
            }
    }
    Eigen::VectorXd p(numActive);
    grids.push_back(outputGrid -> deepCopy());
    grids2.push_back(outputGrid -> deepCopy());
    Eigen::VectorXd b(numActive);        
    Eigen::VectorXd b2(numActive);        
    double error;
    velBeforeUpdate = vels -> deepCopy();
    std::cout << "Before" << std::endl;
    do 
    {

        Eigen::SparseMatrix<double> A(numActive, numActive);    
        Adiag->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        Aplusj->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        Aplusi->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        Aplusk->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        diver->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        rhs->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);


        setRHS(dx, containerGrid, vels, rhs, solidGrid, gravity, dt);
        setDiver(dx, vels, diver, rhs, containerGrid, solidGrid);
        setA(dx, /*dt*/dt, /*pho*/1, containerGrid, Adiag, Aplusi, Aplusj, Aplusk, solidGrid);
        setA2(indices, b, A, diver, Adiag, Aplusi, Aplusj, Aplusk);
        cg.compute(A);
        p = cg.solve(b);
        velUpdate(indices, dx, /*dt*/dt/10, /*pho*/1, vels, containerGrid, p, solidGrid, gravity, defined);

        diver->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        rhs->fill(openvdb::CoordBBox(openvdb::Coord(-60), openvdb::Coord(60)), 0, true);
        setRHS(dx, containerGrid, vels, rhs, solidGrid, gravity, dt);
        setDiver(dx, vels, diver, rhs, containerGrid, solidGrid);
        setOnlyB(indices, b2, diver, Adiag);
        // setA2(indices, b2, A, diver, Adiag, Aplusi, Aplusj, Aplusk);
        error = ((b - b2).norm())/(b.norm());
    }while(error > 0.1);
    pos.defined = defined->deepCopy();
    std::cout << "After" << std::endl;

    // pos.advect(0.1, dx, dt, vels, 60, containerGrid, solidGrid);
    // Same to do for pic
    pos.FLIPadvect(0.1, dx, dt, vels, 60, velBeforeUpdate, containerGrid, solidGrid, normals);
    std::cout << "DT " << dt << std::endl;
    if(i < 500)
    {

        // scatter.operator()<openvdb::FloatGrid>(*fluidGrid);
        // // //containerGrid is dummy here
        // pos.interpFromGrid(vels, 60, containerGrid);
    }
    std::cout << "Error:\t" << error << std::endl;
    std::cout << "Iteration:\t" << i+1 << std::endl;
    simulationTime += dt;
    std::cout << "Time delta:\t" << simulationTime << std::endl;
    file2.write(grids2);
    file2.close();


}
    file.write(grids);
    file.close();

end = clock();
cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
std::cout << "Time Taken " << cpu_time_used/60 << " minutes" << std::endl; 
}

