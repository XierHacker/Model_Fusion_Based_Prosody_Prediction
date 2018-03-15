def get_dataset(images_paths,labels,batch_size,shuffle=True,last_batch=True):
    def _decode_images(file_path, label):
        image_string = tf.read_file(file_path)
        image_decoded = tf.image.decode_png(image_string)
        image = tf.cast(image_decoded, tf.float32) / 255.
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images_paths), tf.constant(labels)))
    dataset = dataset.map(_decode_images)
    # buffer_size = 10 * batch_size
    buffer_size = 50000
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    if not last_batch:
        dataset.filter(lambda x, y: tf.equal(tf.shape(x)[0], batch_size))
    dataset = dataset.batch(batch_size)
    return dataset