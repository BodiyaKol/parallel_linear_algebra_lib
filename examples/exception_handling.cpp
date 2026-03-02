// Приклади використання винятків бібліотеки PLA
// Цей файл показує як ловити та обробляти помилки

#include "../include/main_api.h"
#include <iostream>

using namespace pla;

// Оголошення функцій (після реалізації в src/types)
namespace pla {
    Vector vector_add(const Vector& a, const Vector& b);
    void vector_normalize(Vector& v);
    void matrix_identity(Matrix& m);
    Matrix matrix_multiply(const Matrix& A, const Matrix& B);
}

// ============================================================================
// ПРИКЛАД 1: Обробка невідповідності розмірів векторів
// ============================================================================
void example_vector_size_mismatch() {
    std::cout << "\n=== Приклад 1: SizeMismatchException ===" << std::endl;
    
    try {
        Vector a = {1.0, 2.0, 3.0};
        Vector b = {4.0, 5.0};  // Різні розміри!
        
        // Спроба додати вектори різних розмірів
        Vector c = vector_add(a, b);  // Викине SizeMismatchException
        
    } catch (const SizeMismatchException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: перевір що вектори мають однаковий розмір" << std::endl;
    }
}

// ============================================================================
// ПРИКЛАД 2: Обробка нульового вектора при нормалізації
// ============================================================================
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

// ============================================================================
// ПРИКЛАД 3: Обробка некватратної матриці для identity
// ============================================================================
void example_non_square_matrix() {
    std::cout << "\n=== Приклад 3: NonSquareMatrixException ===" << std::endl;
    
    try {
        Matrix m(3, 4);  // Не квадратна матриця
        
        // Спроба створити identity з некватратної матриці
        matrix_identity(m);  // Викине NonSquareMatrixException
        
    } catch (const NonSquareMatrixException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: використовуй квадратні матриці (rows == cols)" << std::endl;
    }
}

// ============================================================================
// ПРИКЛАД 4: Обробка невідповідності форм при множенні матриць
// ============================================================================
void example_matrix_multiply_mismatch() {
    std::cout << "\n=== Приклад 4: ShapeMismatchException ===" << std::endl;
    
    try {
        Matrix A(2, 3);  // 2x3
        Matrix B(5, 2);  // 5x2 - несумісні розміри!
        
        // Щоб множити A*B потрібно A.cols() == B.rows()
        // Тут 3 != 5, тому викине помилку
        Matrix C = matrix_multiply(A, B);
        
    } catch (const ShapeMismatchException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
        std::cout << "  Рішення: для A*B потрібно щоб A.cols() == B.rows()" << std::endl;
    }
}

// ============================================================================
// ПРИКЛАД 5: Обробка виходу за межі індексу
// ============================================================================
void example_index_out_of_range() {
    std::cout << "\n=== Приклад 5: IndexOutOfRangeException ===" << std::endl;
    
    try {
        Vector v = {1.0, 2.0, 3.0};
        
        // Спроба доступу до неіснуючого елемента
        // Увага: operator[] не кидає виняток, але функції get_row/get_col - кидають
        // Scalar x = v[100];  // Це undefined behavior
        
        // Натомість розглянемо matrix_get_row з невалідним індексом
        Matrix m(2, 3);
        // Vector row = matrix_get_row(m, 5);  // row_index=5 >= rows=2
        
        std::cout << "  (Приклад для matrix_get_row/col коли реалізуєш)" << std::endl;
        
    } catch (const IndexOutOfRangeException& e) {
        std::cout << "  Помилка: " << e.what() << std::endl;
    }
}

// ============================================================================
// ПРИКЛАД 6: Загальна обробка всіх помилок бібліотеки
// ============================================================================
void example_catch_all_pla_exceptions() {
    std::cout << "\n=== Приклад 6: PLAException (базовий клас) ===" << std::endl;
    
    try {
        // Якийсь код що може викинути різні помилки
        Vector a = {1.0, 2.0};
        Vector b = {3.0, 4.0, 5.0};
        
        Vector c = vector_add(a, b);
        
    } catch (const PLAException& e) {
        // Ловимо всі помилки бібліотеки через базовий клас
        std::cout << "  Помилка PLA: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Ловимо інші стандартні помилки
        std::cout << "  Стандартна помилка: " << e.what() << std::endl;
    }
}

// ============================================================================
// ПРИКЛАД 7: Правильна обробка - перевірка перед операцією
// ============================================================================
void example_safe_operations() {
    std::cout << "\n=== Приклад 7: Безпечні операції (перевірка перед викликом) ===" << std::endl;
    
    Vector a = {1.0, 2.0, 3.0};
    Vector b = {4.0, 5.0};
    
    // ПРАВИЛЬНО: перевірка перед операцією
    if (a.dimension() == b.dimension()) {
        Vector c = a + b;
        std::cout << "  Успішно додано вектори" << std::endl;
    } else {
        std::cout << "  Розміри не співпадають: " << a.dimension() 
                  << " != " << b.dimension() << std::endl;
        std::cout << "  Пропускаємо операцію" << std::endl;
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   ПРИКЛАДИ ОБРОБКИ ПОМИЛОК PLA" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Увага: Ці приклади спрацюють після того як ти реалізуєш функції в src/types
    // Поки що розкоментуй тільки прості приклади
    
    // example_vector_size_mismatch();
    // example_zero_vector();
    // example_non_square_matrix();
    // example_matrix_multiply_mismatch();
    // example_index_out_of_range();
    // example_catch_all_pla_exceptions();
    example_safe_operations();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Кінець прикладів" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
