# Організація репозиторію parallel_linear_algebra_lib

## Загальна структура

```
parallel_linear_algebra_lib/
├── include/              # Публічні заголовки API
├── src/                  # Реалізація
├── test/                 # Тести
├── examples/             # Приклади використання
├── CMakeLists.txt        # Конфігурація збірки
├── README.md             # Опис проекту
└── LICENSE               # Ліцензія
```

---

## 📁 `include/` - Публічні заголовки

Заголовки, які бачить користувач бібліотеки. Тільки оголошення класів, методів та функцій.

### `include/main_api.h`
**Призначення**: Єдиний файл для підключення всієї бібліотеки користувачем.  
**Використання**: `#include "main_api.h"`  
**Що робить**: Агрегує всі підзаголовки з `types/`, `blas1/`, `blas2/`, `blas3/`, `linalg_algorithms/`.

---

### 📂 `include/types/` - Базові типи

Основні типи даних бібліотеки. Цю частину API реалізує студент.

#### `types.h`
Агрегатор всіх типів. Включає всі інші заголовки з `types/`.

#### `scalar.h`
```cpp
using Scalar = double;
```
Базовий числовий тип для всіх обчислень.

#### `index.h`
```cpp
using Index = std::size_t;
```
Тип для індексів та розмірів масивів.

#### `vector.h` ⭐
**Клас `Vector`** - математичний вектор з лінійної алгебри.

**Основні оператори**:
- `operator+`, `operator-`, `operator*`, `operator/` - арифметика
- `operator+=`, `operator-=`, `operator*=`, `operator/=` - складені оператори

**Методи**:
- `dimension()` - розмірність вектора (належність до $ \mathbb{R}^n $)
- `coordinates()` - доступ до масиву координат
- `dot(const Vector&)` - скалярний добуток
- `norm()` - Евклідова норма
- `normalize()` - нормалізація in-place
- `normalized()` - повертає нормалізовану копію
- `is_unit()` - перевірка, чи є одиничним

**Конструктори та Rule of 5**:
- `Vector()` - порожній вектор
- `Vector(Index n)` - вектор розмірності n
- `Vector(Index n, Scalar value)` - заповнений значенням
- `Vector(std::initializer_list<Scalar>)` - з списку `{1.0, 2.0, 3.0}`
- Копіюючий конструктор (default)
- Переміщуючий конструктор (default)
- Копіюючий оператор присвоєння (default)
- Переміщуючий оператор присвоєння (default)
- Деструктор (default)

**Приклад**:
```cpp
Vector a = {1.0, 2.0, 3.0};
Vector b = {4.0, 5.0, 6.0};
Vector c = a + b;              // {5.0, 7.0, 9.0}
Scalar dot = a.dot(b);         // 32.0
Scalar norm = a.norm();        // 3.74
Index dim = a.dimension();     // 3 (вектор у R^3)
```

#### `matrix.h` ⭐
**Клас `Matrix`** - математична матриця з лінійної алгебри.

**Основні оператори**:
- `operator+`, `operator-`, `operator*`, `operator/` - арифметика
- `operator*(const Vector&)` - множення матриці на вектор (Ax)
- `operator*(const Matrix&)` - множення матриць (AB)
- `operator+=`, `operator-=`, `operator*=`, `operator/=` - складені оператори

**Методи**:
- `rows()`, `cols()` - розміри матриці
- `data()` - доступ до масиву елементів
- `transpose()` - транспонування (повертає нову матрицю)
- `transpose_inplace()` - транспонування на місці
- `row(Index)` - отримати рядок як `VectorView`
- `col(Index)` - отримати стовпець як `VectorView`
- `fill(Scalar)` - заповнити значенням
- `set_identity()` - зробити одиничною
- `norm()` - норма Фробеніуса
- `static identity(Index)` - створити одиничну матрицю

**Конструктори та Rule of 5**:
- `Matrix()` - порожня матриця
- `Matrix(Index rows, Index cols)` - заданого розміру
- `Matrix(Index rows, Index cols, Scalar value)` - заповнена значенням
- `Matrix(Index rows, Index cols, Scalar value, StorageOrder)` - з вказаним порядком
- Копіюючий конструктор (default)
- Переміщуючий конструктор (default)
- Копіюючий оператор присвоєння (default)
- Переміщуючий оператор присвоєння (default)
- Деструктор (default)

**Приклад**:
```cpp
Matrix A(2, 3);
A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;

Vector x = {1.0, 2.0, 3.0};
Vector y = A * x;              // Матриця × Вектор: {14.0, 32.0}

Matrix B = A.transpose();      // Транспонування: 3×2
Matrix I = Matrix::identity(3); // Одинична матриця 3×3
```

#### `layout.h`
```cpp
enum class StorageOrder { RowMajor, ColMajor };
```
Порядок зберігання елементів матриці в пам'яті:
- `RowMajor` - по рядках (C-style, за замовчуванням)
- `ColMajor` - по стовпцях (Fortran-style, для BLAS/LAPACK)

#### `shape.h`
Структури для опису форми векторів та матриць:
```cpp
struct VectorShape { Index size; };
struct MatrixShape { Index rows; Index cols; };
```

#### `status.h`
Enum та структура для повернення статусів операцій:
```cpp
enum class StatusCode { Ok, InvalidSize, SizeMismatch, ... };
struct Status { StatusCode code; std::string message; };
```

#### `exceptions.h`
Ієрархія винятків для обробки помилок:
- `PLAException` - базовий клас
- `SizeMismatchException` - розміри векторів не співпадають
- `ShapeMismatchException` - форми матриць несумісні
- `ZeroVectorException` - нульовий вектор (нормалізація)
- `NonSquareMatrixException` - потрібна квадратна матриця
- `IndexOutOfRangeException` - вихід за межі
- `SingularMatrixException` - сингулярна матриця
- `InvalidScalarException` - некоректне скалярне значення
- `AllocationException` - помилка виділення пам'яті
- `ConvergenceException` - не збіглося в ітеративному алгоритмі
- `NotImplementedException` - функція не реалізована

#### `view.h`
Неволодіючі view на дані (zero-copy):
```cpp
class VectorView;  // Доступ до частини вектора без копіювання
class MatrixView;  // Доступ до частини матриці без копіювання
```

#### `execution_policy.h`
Налаштування паралелізації:
```cpp
enum class Backend { Serial, Simd, Tbb };
struct ExecutionPolicy { Backend backend; Index thread_count; Index block_size; };
```

---

### 📂 `include/blas1/`
Заголовки для операцій BLAS Level 1 (вектор-вектор).  
Приклади: `axpy` (y = αx + y), `dot` (скалярний добуток), `nrm2` (норма).

### 📂 `include/blas2/`
Заголовки для операцій BLAS Level 2 (матриця-вектор).  
Приклади: `gemv` (y = αAx + βy), `ger` (A = αxy^T + A).

### 📂 `include/blas3/`
Заголовки для операцій BLAS Level 3 (матриця-матриця).  
Приклад: `gemm` (C = αAB + βC).

### 📂 `include/linalg_algorithms/`
Заголовки для алгоритмів лінійної алгебри.  
Приклади: розв'язання систем рівнянь, власні значення, SVD, QR-розклад.

---

## 📁 `src/` - Реалізація

Тут знаходяться файли `.cpp` з реалізацією методів.

### 📂 `src/api/`

#### `main_api.cpp`
Допоміжні функції API, якщо потрібні. Може бути порожнім.

---

### 📂 `src/types/` ⭐ (Основна робоча папка студента)

#### `vector.cpp`
**Реалізація всіх методів класу `Vector`.**

Містить 14 TODO для реалізації:
1. `Vector::operator+` - додавання векторів
2. `Vector::operator-` (бінарний) - віднімання векторів
3. `Vector::operator-` (унарний) - зміна знаку
4. `Vector::operator*` - множення на скаляр
5. `Vector::operator/` - ділення на скаляр
6. `operator*` (зовнішній) - `scalar * vector`
7. `Vector::operator+=` - додавання з присвоєнням
8. `Vector::operator-=` - віднімання з присвоєнням
9. `Vector::operator*=` - множення на скаляр з присвоєнням
10. `Vector::operator/=` - ділення на скаляр з присвоєнням
11. `Vector::dot` - скалярний добуток
12. `Vector::norm` - Евклідова норма
13. `Vector::normalize` - нормалізація in-place
14. `Vector::normalized` - повертає нормалізовану копію

**Структура кожної функції**:
```cpp
Vector Vector::operator+(const Vector& other) const {
    // TODO: Step 1 - перевірка розмірів
    if (dimension() != other.dimension()) {
        throw SizeMismatchException(dimension(), other.dimension());
    }
    
    // TODO: Step 2 - створити результат
    Vector result(dimension());
    
    // TODO: Step 3 - обчислити поелементно
    for (Index i = 0; i < dimension(); i++) {
        result[i] = coordinates_[i] + other.coordinates_[i];
    }
    
    // TODO: Step 4 - повернути результат
    return result;
}
```

#### `matrix.cpp`
**Реалізація всіх методів класу `Matrix`.**

Містить 20 TODO для реалізації:
1. `Matrix::operator+` - додавання матриць
2. `Matrix::operator-` (бінарний) - віднімання матриць
3. `Matrix::operator-` (унарний) - зміна знаку
4. `Matrix::operator*` (скаляр) - множення на скаляр
5. `Matrix::operator/` - ділення на скаляр
6. `Matrix::operator*` (Vector) - **матриця × вектор** (BLAS Level 2)
7. `Matrix::operator*` (Matrix) - **матриця × матриця** (BLAS Level 3)
8. `operator*` (зовнішній) - `scalar * matrix`
9. `Matrix::operator+=` - додавання з присвоєнням
10. `Matrix::operator-=` - віднімання з присвоєнням
11. `Matrix::operator*=` - множення на скаляр з присвоєнням
12. `Matrix::operator/=` - ділення на скаляр з присвоєнням
13. `Matrix::transpose` - транспонування (нова матриця)
14. `Matrix::transpose_inplace` - транспонування на місці
15. `Matrix::row` - отримати рядок
16. `Matrix::col` - отримати стовпець
17. `Matrix::fill` - заповнити значенням
18. `Matrix::set_identity` - зробити одиничною
19. `Matrix::identity` (static) - створити одиничну матрицю
20. `Matrix::norm` - норма Фробеніуса

**Найскладніші операції**:
- **Matrix × Vector**: O(mn) складність, BLAS Level 2
- **Matrix × Matrix**: O(mnp) складність, BLAS Level 3

---

## 📁 `test/` - Тести

### 📂 `test/types/`

#### `test_types.cpp`
Тести для класів `Vector` та `Matrix`.

**Базові тести** (працюють одразу):
- `test_vector_basics()` - конструктори, індексація
- `test_matrix_basics()` - конструктори, індексація
- `test_shape_and_status()` - перевірка допоміжних структур
- `test_execution_policy_defaults()` - перевірка налаштувань за замовчуванням

**Тести операторів** (розкоментуй після реалізації):
- `test_vector_operators_add()` - тест `Vector::operator+`
- `test_vector_operators_multiply()` - тест `Vector::operator*` зі скаляром
- `test_vector_dot()` - тест скалярного добутку
- `test_vector_norm()` - тест норми
- `test_matrix_identity_static()` - тест `Matrix::identity()`
- `test_matrix_transpose()` - тест `Matrix::transpose()`
- `test_matrix_vector_multiply()` - тест `A * x`
- `test_matrix_multiply()` - тест `A * B`

**Як тестувати**:
1. Реалізуй оператор/метод у `vector.cpp` або `matrix.cpp`
2. Розкоментуй відповідний тест
3. Збери проект: `cmake -B build && cmake --build build`
4. Запусти тести: `./build/test_types` або `ctest --test-dir build`

---

## 📁 `examples/` - Приклади використання

### `basic_usage.cpp`
Показує як користуватися бібліотекою:
- Створення векторів і матриць
- Операції з операторами: `a + b`, `A * x`, `A * B`
- Скалярний добуток, норма, транспонування
- Складені оператори: `a += b`, `a *= 2.0`

**Запуск**:
```bash
g++ -std=c++17 -I./include examples/basic_usage.cpp ./build/libpla.a -o basic_usage
./basic_usage
```

### `exception_handling.cpp`
Показує обробку помилок через винятки:
- `SizeMismatchException` - додавання векторів різних розмірів
- `ZeroVectorException` - нормалізація нульового вектора
- `ShapeMismatchException` - множення несумісних матриць
- `IndexOutOfRangeException` - вихід за межі індексу

---

## 📄 Кореневі файли

### `CMakeLists.txt`
Конфігурація збірки проекту.

**Побудова бібліотеки**:
- Створює статичну бібліотеку `libpla.a`
- Компілює `src/api/main_api.cpp`, `src/types/vector.cpp`, `src/types/matrix.cpp`
- Стандарт: C++17

**Збірка тестів**:
- Виконуваний файл `test_types`
- Лінкується з `libpla`

**Команди збірки**:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

### `README.md`
Опис проекту, інструкції з використання.

### `LICENSE`
Ліцензія проекту (MIT, GPL, тощо).

### `.gitignore`
Список файлів/папок, які git ігнорує:
- `build/` - результати збірки
- `*.o`, `*.a` - об'єктні файли та бібліотеки
- IDE налаштування (`.vscode/`, `.idea/`, тощо)

---

## 🎯 Порядок роботи над проектом

### Фаза 1: Прості операції (почни тут!)
1. ✅ `Vector::operator+` - поелементне додавання
2. ✅ `Vector::operator*` (скаляр) - множення кожного елементу
3. ✅ `Vector::dot` - сума добутків елементів
4. ✅ `Vector::norm` - корінь із суми квадратів
5. ✅ `Matrix::fill` - проста заповнення циклом
6. ✅ `Matrix::operator+` - поелементне додавання

Розкоментуй тести після кожної реалізації!

### Фаза 2: Середні операції
7. ✅ `Vector::normalize` - ділення на норму (перевір на нуль!)
8. ✅ `Matrix::transpose` - копіювання з перестановкою індексів
9. ✅ `Matrix::identity` - заповнити діагональ одиницями
10. ✅ Всі складені оператори `+=`, `-=`, `*=`, `/=`

### Фаза 3: Складні операції (залиш на кінець!)
11. ✅ **Matrix × Vector** - O(mn) алгоритм, BLAS Level 2
12. ✅ **Matrix × Matrix** - O(mnp) алгоритм, BLAS Level 3 (найскладніше!)

---

## 🔗 Зв'язки між файлами

```
Користувач
    ↓
#include "main_api.h"
    ↓
main_api.h → types/types.h → vector.h, matrix.h, exceptions.h, ...
    ↓                              ↓
    ↓                         vector.cpp
    ↓                         matrix.cpp
    ↓
Лінкування з libpla.a
```

**Ланцюжок компіляції**:
1. Користувач підключає `main_api.h`
2. `main_api.h` підключає `types/types.h`
3. `types.h` підключає всі заголовки з `types/`
4. Заголовки містять оголошення класів `Vector`, `Matrix`
5. Реалізація в `vector.cpp`, `matrix.cpp` компілюється в `libpla.a`
6. Користувач лінкується з `libpla.a`

---

## 📚 Додаткові ресурси

- `GUIDE_UK.md` - детальний посібник українською мовою
- `REDESIGN_SUMMARY.md` - підсумок редизайну на оператори
- Документація BLAS: https://netlib.org/blas/
- Документація LAPACK: https://netlib.org/lapack/

---

**Успіхів у реалізації! 🚀**
