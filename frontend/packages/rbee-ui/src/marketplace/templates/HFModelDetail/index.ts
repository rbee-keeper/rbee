// TEAM-463: Renamed from ModelDetailPageTemplate to HuggingFaceModelDetail
// TEAM-464: Fixed typo and added backward-compatible export

export type {
  HFModelDetailData,
  HFModelDetailData as ModelDetailData,
  HFModelDetailPageTemplateProps,
  HFModelDetailPageTemplateProps as ModelDetailPageTemplateProps,
} from './HFModelDetail'
export { HFModelDetail, HFModelDetail as HuggingFaceModelDetail, HFModelDetail as ModelDetailPageTemplate } from './HFModelDetail'
