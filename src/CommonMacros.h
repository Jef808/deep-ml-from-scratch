/**
 * @file CommonMacros.h
 *
 * @brief This file contains common macros for generating class member getter functions.
 */

#ifndef COMMONMACROS_H
#define COMMONMACROS_H

/**
 * @brief Defines a constant getter function for a class member.
 * 
 * This macro generates a const-qualified getter member function for a class,
 * providing read-only access to a specified private class member.
 * 
 * @param Type The type of the class member for which the getter is defined.
 * @param Name The name of the class member (without the 'm_' prefix) for which the getter is defined.
 * 
 * Usage:
 * Inside a class definition, use DEFINE_CONST_GETTER(Type, MemberName) where
 * Type is the type of the member and MemberName is the name of the member
 * variable without the 'm_' prefix. This macro expands to a function definition
 * that returns a constant reference to the member variable.
 * 
 * Example:
 * If you have a private member `std::vector<double> m_data;` in your class,
 * you can define a getter for it as follows:
 *
 * @code
 * DEFINE_CONST_GETTER(std::vector<double>, data)
 * @endcode
 *
 * This expands to:
 *
 * @code
 * const std::vector<double>& data() const {
 *     return m_data;
 * }
 * @endcode
 */
#define DEFINE_CONST_GETTER(Type, Name) \
const Type& Name() const { \
  return m_##Name; \
}

/**
 * @Defines a non-const getter function for a class member.
 *
 * This macro generates a non-const-qualified getter member function for a class,
 * providing read-write access to a specified private class member.
 *
 * @param Type The type of the class member for which the getter is defined.
 * @param Name The name of the class member (without the 'm_' prefix) for which the getter is defined.
 *
 * Usage:
 * Inside a class definition, use DEFINE_GETTER(Type, MemberName) where
 * Type is the type of the member and MemberName is the name of the member
 * variable without the 'm_' prefix. This macro expands to a function definition
 * that returns a reference to the member variable.
 *
 * Example:
 * If you have a private member `std::vector<double> m_data;` in your class,
 * you can define a getter for it as follows:
 *
 * @code
 *   DEFINE_GETTER(std::vector<double>, data)
 * @endcode
 *
 * This expands to:
 *
 * @code
 * std::vector<double>& data() {
 *   return m_data;
 * }
 * @endcode
 */
#define DEFINE_GETTER(Type, Name) \
Type& Name() { \
  return m_##Name; \
}

/**
 * @brief Macro to define a default constructor within a class.
 *
 * This macro simplifies the definition of a default constructor for a class by
 * automatically generating a public default constructor that is defaulted. Using this
 * macro within a class definition enhances readability and reduces boilerplate code,
 * especially in cases where classes are intended to be simple or are required to have
 * a default constructor for compliance with certain C++ standards or practices.
 *
 * Usage:
 * Inside a class definition, simply use `DEFAULT_CTOR(ClassName);` to declare and define a
 * default constructor. Ensure that this macro is placed in the public section of the
 * class to adhere to C++ best practices regarding constructor accessibility.
 *
 * Example:
 * ```cpp
 * class MyClass {
 * public:
 *   DEFINE_DEFAULT_CTOR(MyClass);
 * };
 * ```
 */
#define DEFINE_DEFAULT_CTOR(ClassName) \
  ClassName() = default;

#endif /* COMMONMACROS_H */
