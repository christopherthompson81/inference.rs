using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using InferenceRs.AvaloniaDemo.Services;
using InferenceRs.AvaloniaDemo.ViewModels;
using InferenceRs.AvaloniaDemo.Views;

namespace InferenceRs.AvaloniaDemo;

public sealed class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var window = new MainWindow();
            var main = new MainViewModel(new FileService(window));
            window.DataContext = main;
            // engines hold GPU memory and native threads, so they go with the window
            window.Closed += (_, _) => main.Dispose();
            desktop.MainWindow = window;
        }
        base.OnFrameworkInitializationCompleted();
    }
}
