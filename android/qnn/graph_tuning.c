// Research adapter over the public QNN API; one real, version-matched HTP backend.
// Lets ORT use documented HTP graph options its provider does not expose.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "QnnInterface.h"
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpGraph.h"
#ifdef __ANDROID__
#include <android/log.h>
#define TUNE_LOG(...) __android_log_print(ANDROID_LOG_INFO, "JetlinkQnnTune", __VA_ARGS__)
#else
#define TUNE_LOG(...) ((void)0)
#endif

static pthread_once_t initialized = PTHREAD_ONCE_INIT;
static void *backend;
static QnnInterface_t wrapped;
static const QnnInterface_t *provider_list[] = {&wrapped};
static QnnGraph_CreateFn_t real_create;
static QnnContext_FreeFn_t real_context_free;
static Qnn_ErrorHandle_t init_status = QNN_COMMON_ERROR_GENERAL;

// Keep our configuration storage valid for the whole context lifetime. This
// also accommodates backends that consume a setting during graph finalization.
typedef struct ConfigStorage {
  struct ConfigStorage *next;
  Qnn_ContextHandle_t context;
  QnnHtpGraph_CustomConfig_t custom[4];
  QnnGraph_Config_t config[4];
  const QnnGraph_Config_t *all[];
} ConfigStorage;
static ConfigStorage *retained;
static pthread_mutex_t retained_mutex = PTHREAD_MUTEX_INITIALIZER;

static Qnn_ErrorHandle_t tuned_context_free(Qnn_ContextHandle_t context,
    Qnn_ProfileHandle_t profile) {
  Qnn_ErrorHandle_t status = real_context_free(context, profile);
  if (status != QNN_SUCCESS) return status;
  pthread_mutex_lock(&retained_mutex);
  ConfigStorage **link = &retained;
  while (*link != NULL) {
    ConfigStorage *entry = *link;
    if (entry->context == context) {
      *link = entry->next;
      free(entry);
    } else {
      link = &entry->next;
    }
  }
  pthread_mutex_unlock(&retained_mutex);
  return status;
}

static Qnn_ErrorHandle_t tuned_create(Qnn_ContextHandle_t context, const char *name,
    const QnnGraph_Config_t **original, Qnn_GraphHandle_t *graph) {
  const char *tuning = getenv("JETLINK_QNN_TUNING");
  TUNE_LOG("graphCreate graph=%s tuning=%s", name ? name : "", tuning ? tuning : "stock");
  if (tuning == NULL || strcmp(tuning, "stock") == 0) {
    return real_create(context, name, original, graph);
  }
  bool memory = strcmp(tuning, "memory") == 0 || strcmp(tuning, "memory-accurate") == 0;
  bool accurate = strcmp(tuning, "accurate") == 0 || strcmp(tuning, "memory-accurate") == 0;
  bool no_fusion = accurate || strcmp(tuning, "no-fusion") == 0;
  bool no_short_hmx = accurate || strcmp(tuning, "no-short-hmx") == 0;
  bool high_sigmoid = accurate || strcmp(tuning, "high-sigmoid") == 0;
  if (!memory && !no_fusion && !no_short_hmx && !high_sigmoid) {
    return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  }
  size_t n_original = 0;
  if (original != NULL) {
    while (n_original < 128 && original[n_original] != NULL) ++n_original;
    if (n_original == 128) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  }
  ConfigStorage *storage = calloc(1, sizeof(*storage) + (n_original + 5) * sizeof(*storage->all));
  if (storage == NULL) return QNN_COMMON_ERROR_MEM_ALLOC;
  storage->context = context;
  const QnnGraph_Config_t **all = storage->all;
  for (size_t i=0; i<n_original; ++i) all[i]=original[i];
  QnnHtpGraph_CustomConfig_t *custom = storage->custom;
  QnnGraph_Config_t *config = storage->config;
  size_t count = 0;
#define ADD_CONFIG(OPTION, FIELD, VALUE) do { \
    custom[count].option = OPTION; custom[count].FIELD = VALUE; \
    config[count].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM; \
    config[count].customConfig = &custom[count]; \
    all[n_original + count] = &config[count]; ++count; \
  } while (0)
  if (memory) ADD_CONFIG(QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE, vtcmSizeInMB, QNN_HTP_GRAPH_CONFIG_OPTION_MAX);
  if (no_fusion) ADD_CONFIG(QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION, advancedActivationFusion, false);
  if (no_short_hmx) ADD_CONFIG(QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF, shortDepthConvOnHmxOff, true);
  if (high_sigmoid) ADD_CONFIG(QNN_HTP_GRAPH_CONFIG_OPTION_HIGH_PRECISION_SIGMOID, highPrecisionSigmoid, true);
#undef ADD_CONFIG
  Qnn_ErrorHandle_t status = real_create(context, name, all, graph);
  TUNE_LOG("graphCreate added=%zu status=%llu", count, (unsigned long long)status);
  if (status == QNN_SUCCESS) {
    pthread_mutex_lock(&retained_mutex);
    storage->next = retained;
    retained = storage;
    pthread_mutex_unlock(&retained_mutex);
  } else {
    free(storage);
  }
  return status;
}

static void initialize(void) {
  Dl_info location;
  if (dladdr((void *)&initialize, &location) == 0 || location.dli_fname == NULL) return;
  const char *slash = strrchr(location.dli_fname, '/');
  if (slash == NULL) return;
  char path[4096];
  if (snprintf(path, sizeof(path), "%.*s/libQnnHtp.so", (int)(slash-location.dli_fname), location.dli_fname) >= (int)sizeof(path)) return;
  backend = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (backend == NULL) {
    fprintf(stderr, "Jetlink QNN adapter: cannot open the matched HTP backend: %s\n", dlerror());
    return;
  }
  typedef Qnn_ErrorHandle_t (*GetProviders)(const QnnInterface_t ***, uint32_t *);
  GetProviders get = (GetProviders)dlsym(backend, "QnnInterface_getProviders");
  if (get == NULL) return;
  const QnnInterface_t **providers = NULL;
  uint32_t count = 0;
  if (get(&providers, &count) != QNN_SUCCESS) return;
  for (uint32_t i=0; i<count; ++i) {
    const QnnInterface_t *p = providers[i];
    if (p == NULL || p->backendId != QNN_BACKEND_ID_HTP ||
        p->apiVersion.coreApiVersion.major != QNN_API_VERSION_MAJOR ||
        p->apiVersion.coreApiVersion.minor < QNN_API_VERSION_MINOR) continue;
    wrapped = *p;
    real_create = p->QNN_INTERFACE_VER_NAME.graphCreate;
    real_context_free = p->QNN_INTERFACE_VER_NAME.contextFree;
    if (real_create == NULL || real_context_free == NULL) return;
    wrapped.QNN_INTERFACE_VER_NAME.graphCreate = tuned_create;
    wrapped.QNN_INTERFACE_VER_NAME.contextFree = tuned_context_free;
    const char *build_id = "unavailable";
    if (p->QNN_INTERFACE_VER_NAME.backendGetBuildId != NULL) {
      p->QNN_INTERFACE_VER_NAME.backendGetBuildId(&build_id);
    }
    TUNE_LOG("backend core=%u.%u build=%s", p->apiVersion.coreApiVersion.major,
             p->apiVersion.coreApiVersion.minor, build_id ? build_id : "unavailable");
    init_status = QNN_SUCCESS;
    return;
  }
}

Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t ***providers, uint32_t *count) {
  if (providers == NULL || count == NULL) return QNN_COMMON_ERROR_INVALID_ARGUMENT;
  pthread_once(&initialized, initialize);
  if (init_status != QNN_SUCCESS) return init_status;
  *providers = provider_list;
  *count = 1;
  return QNN_SUCCESS;
}
