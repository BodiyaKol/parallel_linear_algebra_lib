#include <pla/pla.h>
#include <iostream>

using namespace pla;

namespace pla {
    Vector vector_add(const Vector& a, const Vector& b);
    void vector_normalize(Vector& v);
    void matrix_identity(Matrix& m);
    Matrix matrix_multiply(const Matrix& A, const Matrix& B);
}

void example_vector_size_mismatch() {
    std::cout << "\n=== Приклад 1: SizeMismatchException ===" << std::endl;
    
    try {
        Vector a = {1.0, 2.0, 3.0};
        Vector b = {4.0, 5.0};  // Різні розміри!
        
        Vector c = vector_add(a, b);  // Викине SizeMismatchException
        
    } catch (const SizeMismatchException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: перевір що вектори мають однаковий розмір" << std::endl;
    }
}


void example_zero_vector() {
    std::cout << "\n=== Приклад 2: ZeroVectorException ===" << std::endl;
    
    try {
        Vector v = {0.0, 0.0, 0.0};  // Нульовий вектор
        
        // Спроба нормалізувати нульовий вектор
        vector_normalize(v);  // Викине ZeroVectorException
        
    } catch (const ZeroVectorException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: перевір що вектор не нульовий перед нормалізацією" << std::endl;
    }
}


void example_non_square_matrix() {
    std::cout << "\n=== Приклад 3: NonSquareMatrixException ===" << std::endl;
    
    try {
        Matrix m(3, 4);  // Не квадратна матриця
        
        matrix_identity(m);  // Викине NonSquareMatrixException
        
    } catch (const NonSquareMatrixException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: використовуй квадратні матриці (rows == cols)" << std::endl;
    }
}


void example_matrix_multiply_mismatch() {
    std::cout << "\n=== Приклад 4: ShapeMismatchException ===" << std::endl;
    
    try {
        Matrix A(2, 3);
        Matrix B(5, 2);
        
        Matrix C = matrix_multiply(A, B);
        
    } catch (const ShapeMismatchException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: для A*B потрібно щоб A.cols() == B.rows()" << std::endl;
    }
}


void example_index_out_of_range() {
    std::cout << "\n=== Приклад 5: IndexOutOfRangeException ===" << std::endl;
    
    try {
        Vector v = {1.0, 2.0, 3.0};
        
        Matrix m(2, 3);
        
        std::cout << "  (Приклад для matrix_get_row/col коли реалізуєш)" << std::endl;
        
    } catch (const IndexOutOfRangeException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
    }
}


void example_catch_all_pla_exceptions() {
    std::cout << "\n=== Приклад 6: PLAException (базовий клас) ===" << std::endl;
    
    try {
        Vector a = {1.0, 2.0};
        Vector b = {3.0, 4.0, 5.0};
        
        Vector c = vector_add(a, b);
        
    } catch (const PLAException& e) {
        std::cout << "  Помилка PLA: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Стандартна помилка: " << e.what() << std::endl;
    }
}


void example_safe_operations() {
    std::cout << "\n=== Приклад 7: Безпечні операції (перевірка перед викликом) ===" << std::endl;
    
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0};
    
    if (a.dimension() == b.dimension()) {
        Vector c = a + b;
        std::cout << "  Успішно додано вектори" << std::endl;
    } else {
        std::cout << "  Розміри не співпадають: " << a.dimension() 
                  << " != " << b.dimension() << std::endl;
        std::cout << "  Пропускаємо операцію" << std::endl;
    }
}


int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   ПРИКЛАДИ ОБРОБКИ ПОМИЛОК PLA" << std::endl;
    std::cout << "========================================" << std::endl;
    
    example_safe_operations();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Кінець прикладів" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
