use std::sync::Arc;

use image::DynamicImage;
use inference_tensor::Tensor;

use crate::Response;
use crate::sequence::{Sequence, SequenceState, StopReason};

pub async fn send_image_responses(
    input_seqs: &mut [&mut Sequence],
    images: Vec<DynamicImage>,
) -> inference_tensor::Result<()> {
    if input_seqs.len() != images.len() {
        inference_tensor::bail!(
            "Input seqs len ({}) does not match images generated len ({})",
            input_seqs.len(),
            images.len()
        );
    }

    for (seq, image) in input_seqs.iter_mut().zip(images) {
        seq.add_image_to_group(image);
        let created = seq.creation_time() as u128;
        let responder = seq.responder();
        seq.get_mut_group()
            .maybe_send_image_gen_response(created, responder)
            .await
            .map_err(inference_tensor::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedImage));
    }

    Ok(())
}

pub async fn send_transcription_responses(
    input_seqs: &mut [&mut Sequence],
    transcripts: Vec<Result<inference_models_speech::Transcription, String>>,
) -> inference_tensor::Result<()> {
    if input_seqs.len() != transcripts.len() {
        inference_tensor::bail!(
            "Input seqs len ({}) does not match transcripts len ({})",
            input_seqs.len(),
            transcripts.len()
        );
    }
    for (seq, transcript) in input_seqs.iter_mut().zip(transcripts) {
        let response = match transcript {
            Ok(transcript) => Response::Transcription(transcript),
            Err(error) => Response::ValidationError(error.into()),
        };
        // a client gone before its answer must not cost the rest of the batch theirs
        if seq.responder().send(response).await.is_err() {
            tracing::warn!("a transcription's receiver disconnected");
        }
        seq.set_state(SequenceState::Done(StopReason::Transcribed));
    }
    Ok(())
}

pub async fn send_speech_responses(
    input_seqs: &mut [&mut Sequence],
    pcms: &[Arc<Vec<f32>>],
    rates: &[usize],
    channels: &[usize],
) -> inference_tensor::Result<()> {
    if input_seqs.len() != pcms.len() {
        inference_tensor::bail!(
            "Input seqs len ({}) does not match pcms generated len ({})",
            input_seqs.len(),
            pcms.len()
        );
    }

    for (seq, (pcm, (rate, channel))) in input_seqs
        .iter_mut()
        .zip(pcms.iter().zip(rates.iter().zip(channels)))
    {
        seq.add_speech_pcm_to_group(pcm.clone(), *rate, *channel);

        let group = seq.get_mut_group();
        group
            .maybe_send_speech_response(seq.responder())
            .await
            .map_err(inference_tensor::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::GeneratedSpeech));
    }

    Ok(())
}

pub async fn send_raw_responses(
    input_seqs: &mut [&mut Sequence],
    logits_chunks: Vec<Vec<Tensor>>,
) -> inference_tensor::Result<()> {
    let logits_chunks = if logits_chunks.len() == 1 {
        logits_chunks[0].clone()
    } else {
        inference_tensor::bail!("Raw response only supports batch size of 1.");
    };
    assert_eq!(input_seqs.len(), 1);

    let seq = &mut *input_seqs[0];

    seq.add_raw_choice_to_group(logits_chunks);

    let group = seq.get_mut_group();
    group
        .maybe_send_raw_done_response(seq.responder())
        .await
        .map_err(inference_tensor::Error::msg)?;

    seq.set_state(SequenceState::Done(StopReason::Length(0)));

    Ok(())
}

pub async fn send_embedding_responses(
    input_seqs: &mut [&mut Sequence],
    embedings: Vec<Vec<f32>>,
) -> inference_tensor::Result<()> {
    if embedings.len() != input_seqs.len() {
        inference_tensor::bail!("Number of embeddings must match number of sequences..");
    }

    for (seq, embeddings) in input_seqs.iter_mut().zip(embedings) {
        seq.add_embedding_choice_to_group(embeddings);

        let group = seq.get_mut_group();
        group
            .maybe_send_embedding_done_response(seq.responder())
            .await
            .map_err(inference_tensor::Error::msg)?;

        seq.set_state(SequenceState::Done(StopReason::Length(0)));
    }

    Ok(())
}
