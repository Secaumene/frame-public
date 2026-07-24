// 数据类型系统的实现单元。
// 注意落点:itemsize()/name() 是 constexpr 成员(隐式 inline)、dispatch_dtype
// 是函数模板——三者的**定义必须写在 include/frame/core/dtype.h 内**(使用处须
// 可见,写进本 .cpp 会导致其他翻译单元未定义引用/无法常量求值)。
// 本文件仅承接未来真正的非内联符号(如运行期查表数据),当前为空实现单元。

#include <frame/core/dtype.h>

// 三者的定义均已落在 include/frame/core/dtype.h 内(见本文件头部注释),
// 本翻译单元当前无需承接任何符号,继续保持为空实现单元。
