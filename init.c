#include "TH.h"
#include "luaT.h"

#ifdef _OPENMP
#include "omp.h"
#endif

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

DLL_EXPORT int luaopen_libnnexp(lua_State *L)
{
  return 1;
}