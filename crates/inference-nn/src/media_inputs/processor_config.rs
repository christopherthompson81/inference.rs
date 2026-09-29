use serde::Deserialize;

#[derive(Deserialize, Debug, Default)]
pub struct ProcessorConfig {
    pub chat_template: Option<String>,
    #[serde(alias = "image_seq_length")]
    pub image_seq_len: Option<usize>,
    pub image_break_token: Option<String>,
    pub image_end_token: Option<String>,
    pub image_token: Option<String>,
    pub patch_size: Option<usize>,
    pub spatial_merge_size: Option<usize>,
    pub pixel_shuffle_ratio: Option<f32>,
    pub audio_seq_length: Option<usize>,
    pub video_max_soft_tokens: Option<usize>,
}
