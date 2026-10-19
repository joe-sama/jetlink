#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "QnnInterface.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpGraph.h"

#ifdef FAKE_BACKEND
static const QnnGraph_Config_t **saved[2];
static size_t saved_count;
static void check_configs(const QnnGraph_Config_t **configs) {
  int flags = 0, priority = 0;
  for (size_t i=0; configs && configs[i]; ++i) {
    if (configs[i]->option == QNN_GRAPH_CONFIG_OPTION_PRIORITY) { ++priority; continue; }
    assert(configs[i]->option == QNN_GRAPH_CONFIG_OPTION_CUSTOM);
    const QnnHtpGraph_CustomConfig_t *c = configs[i]->customConfig;
    switch(c->option) {
      case QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE: assert(c->vtcmSizeInMB==0); flags|=1; break;
      case QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION: assert(!c->advancedActivationFusion); flags|=2; break;
      case QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF: assert(c->shortDepthConvOnHmxOff); flags|=4; break;
      case QNN_HTP_GRAPH_CONFIG_OPTION_HIGH_PRECISION_SIGMOID: assert(c->highPrecisionSigmoid); flags|=8; break;
      default: assert(false);
    }
  }
  assert(flags==atoi(getenv("EXPECTED_FLAGS")));
  assert(priority==1);
}
static Qnn_ErrorHandle_t create(Qnn_ContextHandle_t context, const char *name,
    const QnnGraph_Config_t **configs, Qnn_GraphHandle_t *graph) {
  (void)context; (void)name;
  check_configs(configs);
  assert(saved_count < 2);
  saved[saved_count++] = configs;
  *graph=(void *)0x1;
  return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t context_free(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile) {
  (void)context; (void)profile;
  for (size_t i=0; i<saved_count; ++i) check_configs(saved[i]);
  saved_count=0;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t ***list, uint32_t *count) {
  static QnnInterface_t p = QNN_INTERFACE_INIT;
  static const QnnInterface_t *ps[] = {&p};
  p.backendId=QNN_BACKEND_ID_HTP;
  p.apiVersion.coreApiVersion.major=QNN_API_VERSION_MAJOR;
  p.apiVersion.coreApiVersion.minor=QNN_API_VERSION_MINOR;
  p.QNN_INTERFACE_VER_NAME.graphCreate=create;
  p.QNN_INTERFACE_VER_NAME.contextFree=context_free;
  *list=ps; *count=1;
  return QNN_SUCCESS;
}
#else
#include <dlfcn.h>
int main(int argc,char **argv) {
  assert(argc==2);
  void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!lib) { puts(dlerror()); return 1; }
  typedef Qnn_ErrorHandle_t (*Get)(const QnnInterface_t ***,uint32_t *);
  Get get=(Get)dlsym(lib,"QnnInterface_getProviders");
  assert(get);
  assert(get(NULL,NULL)==QNN_COMMON_ERROR_INVALID_ARGUMENT);
  const QnnInterface_t **p; uint32_t n;
  assert(get(&p,&n)==QNN_SUCCESS && n==1);
  QnnGraph_Config_t priority=QNN_GRAPH_CONFIG_INIT;
  priority.option=QNN_GRAPH_CONFIG_OPTION_PRIORITY;
  const QnnGraph_Config_t *configs[]={&priority,NULL};
  Qnn_GraphHandle_t graph=NULL;
  Qnn_ErrorHandle_t status=p[0]->QNN_INTERFACE_VER_NAME.graphCreate(NULL,"test",configs,&graph);
  if(strcmp(getenv("JETLINK_QNN_TUNING"),"invalid")==0) assert(status==QNN_COMMON_ERROR_INVALID_ARGUMENT);
  else {
    assert(status==QNN_SUCCESS && graph==(void *)0x1);
    assert(p[0]->QNN_INTERFACE_VER_NAME.graphCreate(NULL,"second",configs,&graph)==QNN_SUCCESS);
  }
  assert(p[0]->QNN_INTERFACE_VER_NAME.contextFree(NULL,NULL)==QNN_SUCCESS);
  puts("configuration forwarding and deferred lifetime passed");
  return 0;
}
#endif
