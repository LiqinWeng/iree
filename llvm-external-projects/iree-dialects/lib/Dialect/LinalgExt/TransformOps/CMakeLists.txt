add_mlir_library(IREELinalgExtTransformOps
  LinalgExtTransformOps.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREEDialectsTransforms
  MLIRRewrite
  MLIRTransformDialect
  
  IREELinalgExtDialect
  IREELinalgExtTransforms

  MLIRPDLDialect
)
