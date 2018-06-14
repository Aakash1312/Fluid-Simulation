rm $1 2> /dev/null
rm mygrids.vdb 2> /dev/null
g++ -std=c++11 -pthread -O3 -DNDEBUG -I . -I .. -isystem /usr/local/include -isystem /usr/local/include -isystem /usr/local/include -DOPENVDB_USE_GLFW_3 -o $1 $1".cc" -I . \
    -Wl,-rpath,./openvdb/build/lib -L./openvdb ./openvdb/libopenvdb.so.4.0.2 \
    -ldl -lm -lz -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lHalf -Wl,-rpath,/usr/local/lib -L/usr/local/lib -ltbb -Wl,-rpath,/usr/local/lib -L/usr/local/lib -lboost_iostreams -lboost_system  -lrt 
echo "Compiled"
./$1
