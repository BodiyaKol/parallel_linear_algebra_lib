#include <iostream>
#include <pla/pla.h>

int main() {
    using pla::Vector;

    { // VECTOR

        Vector<double> a{1.0, 2.0, 3.0};
        Vector<double> b{4.0, 5.0, 6.0};

        std::cout << "a[0] = " << a[0] << "\n";

        auto c = a + b;
        auto d = a - b;
        auto e = a * 2.0;

        std::cout << "a + b = " << c << "\n";
        std::cout << "a - b = " << d << "\n";
        std::cout << "a * 2 = " << e << "\n";

        double dot = a.dot(b);
        std::cout << "a · b = " << dot << "\n";

        std::cout << "||a|| = " << a.norm() << "\n";

        auto unit = a.normalized();
        std::cout << "normalized a = " << unit << "\n";

        a += b;
        std::cout << "a += b → " << a << "\n";

        std::cout << "is unit: " << unit.is_unit() << "\n";
    
    }

    return 0;
}