using System.Collections.Specialized;
using System.ComponentModel;
using Avalonia.Controls;
using Avalonia.Threading;
using InferenceRs.AvaloniaDemo.ViewModels;

namespace InferenceRs.AvaloniaDemo.Views;

public partial class ChatView : UserControl
{
    // within this many pixels of the end counts as following the stream
    private const double FollowSlack = 48;

    private ChatViewModel? _chat;

    public ChatView()
    {
        InitializeComponent();
        DataContextChanged += (_, _) =>
        {
            if (_chat is not null) _chat.Messages.CollectionChanged -= OnMessagesChanged;
            _chat = DataContext as ChatViewModel;
            if (_chat is not null) _chat.Messages.CollectionChanged += OnMessagesChanged;
        };
    }

    private void OnMessagesChanged(object? sender, NotifyCollectionChangedEventArgs change)
    {
        foreach (var message in change.NewItems?.OfType<ChatMessageViewModel>() ?? [])
        {
            message.PropertyChanged += OnMessageChanged;
        }
        FollowStream(force: true);
    }

    private void OnMessageChanged(object? sender, PropertyChangedEventArgs change) => FollowStream(force: false);

    // keeps the newest tokens in view, unless the user scrolled up to read
    private void FollowStream(bool force)
    {
        var following = Scroller.Offset.Y + Scroller.Viewport.Height >= Scroller.Extent.Height - FollowSlack;
        if (force || following) Dispatcher.UIThread.Post(Scroller.ScrollToEnd, DispatcherPriority.Background);
    }
}
